import math
import torch
import torch.nn.functional as F

# Optional fast Newton-Schulz kernel (CUDA only)
try:
    from flash_muon import fast_newtonschulz as _fast_newtonschulz_impl  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - optional dependency
    _fast_newtonschulz_impl = None


"""
An implementation of the parallel scan operation in PyTorch (Blelloch version).
Please see docs/pscan.ipynb for a detailed explanation of what happens here.
"""

def npo2(len):
    """
    Returns the next power of 2 above len
    """
    return 2 ** math.ceil(math.log2(len))

def pad_npo2(X):
    """
    Pads input length dim to the next power of 2

    Args:
        X : (B, L, D, N)

    Returns:
        Y : (B, npo2(L), D, N)
    """
    len_npo2 = npo2(X.size(1))
    pad_tuple = (0, 0, 0, 0, 0, len_npo2 - X.size(1))
    return F.pad(X, pad_tuple, "constant", 0)

class PScan(torch.autograd.Function):
    @staticmethod
    def pscan(A, X):
        # A : (B, D, L, N)
        # X : (B, D, L, N)
        
        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # up sweep
        Aa = A
        Xa = X
        for _ in range(num_steps-2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)
            
            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]

        if Xa.size(2) == 4:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Aa[:, :, 1].mul_(Aa[:, :, 0])
            
            Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 1]))
            Aa[:, :, 2].mul_(Aa[:, :, 1])
            Xa[:, :, 3].add_(Aa[:, :, 3].mul(Xa[:, :, 2]))
            Aa[:, :, 3].mul_(Aa[:, :, 2])

        elif Xa.size(2) == 2:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            return
        else:
            return

        # down sweep
        Aa = A[:, :, 2**(num_steps-2)-1:L:2**(num_steps-2)]
        Xa = X[:, :, 2**(num_steps-2)-1:L:2**(num_steps-2)]
        
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
        Aa[:, :, 1].mul_(Aa[:, :, 0])
        Xa[:, :, 3].add_(Aa[:, :, 3].mul(Xa[:, :, 2]))
        Aa[:, :, 3].mul_(Aa[:, :, 2])

        for k in range(num_steps-3, -1, -1):
            Aa = A[:, :, 2**k-1:L:2**k]
            Xa = X[:, :, 2**k-1:L:2**k]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)

            Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def pscan_rev(A, X):
        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # up sweep
        Aa = A
        Xa = X
        for _ in range(num_steps-2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)
                    
            Xa[:, :, :, 0].add_(Aa[:, :, :, 0].mul(Xa[:, :, :, 1]))
            Aa[:, :, :, 0].mul_(Aa[:, :, :, 1])

            Aa = Aa[:, :, :, 0]
            Xa = Xa[:, :, :, 0]

        if Xa.size(2) == 4:
            Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 3]))
            Aa[:, :, 2].mul_(Aa[:, :, 3])
            
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 2]))
            Aa[:, :, 1].mul_(Aa[:, :, 2])
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
            Aa[:, :, 0].mul_(Aa[:, :, 1])

        elif Xa.size(2) == 2:
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
            return
        else:
            return

        # down sweep
        Aa = A[:, :, 0:L:2**(num_steps-2)]
        Xa = X[:, :, 0:L:2**(num_steps-2)]
        
        Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 3]))
        Aa[:, :, 2].mul_(Aa[:, :, 3])
        Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
        Aa[:, :, 0].mul_(Aa[:, :, 1])

        for k in range(num_steps-3, -1, -1):
            Aa = A[:, :, 0:L:2**k]
            Xa = X[:, :, 0:L:2**k]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)

            Xa[:, :, :-1, 1].add_(Aa[:, :, :-1, 1].mul(Xa[:, :, 1:, 0]))
            Aa[:, :, :-1, 1].mul_(Aa[:, :, 1:, 0])

    @staticmethod
    def forward(ctx, A_in, X_in):
        """
        Args:
            A_in : (B, L, D, N)
            X_in : (B, L, D, N)

        Returns:
            H : (B, L, D, N)
        """
        L = X_in.size(1)

        if L == npo2(L):
            A = A_in.clone()
            X = X_in.clone()
        else:
            A = pad_npo2(A_in)
            X = pad_npo2(X_in)
        
        A = A.transpose(2, 1)
        X = X.transpose(2, 1)

        PScan.pscan(A, X)

        ctx.save_for_backward(A_in, X)
        
        return X.transpose(2, 1)[:, :L]
    
    @staticmethod
    def backward(ctx, grad_output_in):
        A_in, H = ctx.saved_tensors

        L = grad_output_in.size(1)
        L_padded = npo2(L)

        if L == L_padded:
            grad_output = grad_output_in.clone()
            A = A_in.clone()
        else:
            grad_output = pad_npo2(grad_output_in)
            A = pad_npo2(A_in)

        grad_output = grad_output.transpose(2, 1)
        A = A.transpose(2, 1)

        A_scan = torch.nn.functional.pad(A[:, :, 1:], (0, 0, 0, 1))
        
        PScan.pscan_rev(A_scan, grad_output)

        grad_scan = grad_output
        
        H_prev = torch.nn.functional.pad(H[:, :, :-1], (0, 0, 1, 0))

        grad_A = grad_scan * H_prev
        grad_X = grad_scan

        return grad_A.transpose(2, 1)[:, :L], grad_X.transpose(2, 1)[:, :L]
    
pscan = PScan.apply


class PScanMomentumNS5(torch.autograd.Function):
    
    # Newton-Schulz constants (optimized for 1 iteration)
    NS_A = 3.4445
    NS_B = -4.7750
    NS_C = 2.0315
    
    @staticmethod
    def _newton_schulz_forward_data(G, eps: float = 1e-7):
        """
        Helper function for NS forward pass with intermediate values saved.
        
        Args:
            G: (B, L, D, N) input tensor
            
        Returns:
            output: (B, L, D, N) orthogonalized tensor
            saved_data: tuple of intermediate tensors for backward
        """
        out_dtype = G.dtype
        G_bf16 = G.to(torch.bfloat16)
        eps_bf16 = torch.tensor(eps, dtype=torch.bfloat16, device=G.device)

        # Normalize
        s_sq = torch.sum(G_bf16 * G_bf16, dim=(-2, -1), keepdim=True, dtype=torch.bfloat16)
        s = torch.sqrt(s_sq)
        norm = s + eps_bf16

        X_norm_bf16 = G_bf16 / norm

        # Determine if we need to transpose
        transposed = (G.size(-2) > G.size(-1))
        X_work = X_norm_bf16.transpose(-2, -1) if transposed else X_norm_bf16

        # Newton-Schulz iteration (1 step)
        A_ns = X_work @ X_work.transpose(-2, -1)
        B_ns = PScanMomentumNS5.NS_B * A_ns + PScanMomentumNS5.NS_C * (A_ns @ A_ns)
        X_next = PScanMomentumNS5.NS_A * X_work + B_ns @ X_work

        X_out_bf16 = X_next.transpose(-2, -1) if transposed else X_next
        
        saved_data = (G_bf16, X_norm_bf16, X_work, A_ns, B_ns, s, norm, transposed, out_dtype)
        return X_out_bf16, saved_data

    @staticmethod
    def _newton_schulz_backward_data(gradY, saved_tensors):
        """
        Optimized manual backward for Newton-Schulz.
        """
        (G_bf16, X_norm_bf16, X_work, A_ns, B_ns, s, norm, transposed, out_dtype) = saved_tensors
        a, b, c = PScanMomentumNS5.NS_A, PScanMomentumNS5.NS_B, PScanMomentumNS5.NS_C
        
        gradY_bf16 = gradY.to(torch.bfloat16)

        if transposed:
            gradY_bf16 = gradY_bf16.transpose(-2, -1)

        # Gradient through final output
        grad_direct = a * gradY_bf16 + B_ns.transpose(-2, -1) @ gradY_bf16
        
        # Gradient through B_ns
        grad_B = gradY_bf16 @ X_work.transpose(-2, -1)
        
        # Gradient through A_ns
        grad_A = b * grad_B + c * (
            A_ns.transpose(-2, -1) @ grad_B + grad_B @ A_ns.transpose(-2, -1)
        )
        
        grad_from_A = (grad_A + grad_A.transpose(-2, -1)) @ X_work
        grad_X_work_bf16 = grad_direct + grad_from_A
        
        grad_Xn_bf16 = (
            grad_X_work_bf16.transpose(-2, -1)
            if transposed
            else grad_X_work_bf16
        )

        # Gradient through normalization
        dot = torch.sum(
            G_bf16 * grad_Xn_bf16, dim=(-2, -1), keepdim=True, dtype=torch.bfloat16
        )

        grad_base = grad_Xn_bf16 / norm
        denom = s * norm * norm
        correction = G_bf16 * (dot / denom)
        zero_mask = (s == 0)
        grad_G_bf16 = torch.where(zero_mask, grad_base, grad_base - correction)

        return grad_G_bf16.to(out_dtype)

    @staticmethod
    def forward(ctx, A_in, X_in, alpha, beta):
        """
        Forward pass for Muon-inspired Mamba optimizer.
        
        Equations:
            b_t = alpha * deltaB * u_t
            b_t_ortho = Newton-Schulz(b_t)
            v_t = beta * v_{t-1} + b_t_ortho
            h_t = exp(delta*A) * h_{t-1} + v_t
            y_t = C_t * h_t + D_t * u_t
        
        Args:
            A_in: (B, L, D, N) - exponential decay factors exp(delta*A)
            X_in: (B, L, D, N) - deltaB * u_t 
            alpha: scalar - learning rate scale (hyperparameter, no gradient needed)
            beta: scalar - momentum coefficient (hyperparameter, no gradient needed)
            
        Returns:
            H: (B, L, D, N) - hidden states h_t
        """
        L = A_in.size(1)
        L_padded = npo2(L)
        
        # Convert to float32 for numerical stability
        A_in_fp32 = A_in.to(torch.float32)
        X_in_fp32 = X_in.to(torch.float32)
        
        # Pad if needed
        if L == L_padded:
            A_padded = A_in_fp32.clone()
            X_padded = X_in_fp32.clone()
        else:
            A_padded = pad_npo2(A_in_fp32)
            X_padded = pad_npo2(X_in_fp32)
        
        # Prepare for scan: (B, L, D, N) -> (B, D, L, N)
        A = A_padded.transpose(2, 1)
        BX = X_padded.transpose(2, 1)
        
        # Step 1: b_t = alpha * (deltaB * u_t)
        alpha_BX = alpha * BX  # (B, D, L_padded, N)
        
        # Convert to (B, L_padded, D, N) for Newton-Schulz
        alpha_BX_batch = alpha_BX.transpose(1, 2).contiguous()

        # Step 2: b_t_ortho = Newton-Schulz(b_t)
        if (
            not torch.is_grad_enabled()
            and _fast_newtonschulz_impl is not None
            and alpha_BX_batch.is_cuda
        ):
            # Fast inference path
            leading_shape = alpha_BX_batch.shape[:-2]
            m, n = alpha_BX_batch.shape[-2], alpha_BX_batch.shape[-1]
            flat = alpha_BX_batch.reshape(-1, m, n)
            out_list = [
                _fast_newtonschulz_impl(mat, steps=1) for mat in flat
            ]
            b_ortho_batch = (
                torch.stack(out_list, dim=0)
                .reshape(*leading_shape, m, n)
                .to(alpha_BX_batch.dtype)
            )
            ns_saved_tensors = None
        else:
            # Training path with saved intermediates
            b_ortho_batch_bf16, ns_saved_tensors = PScanMomentumNS5._newton_schulz_forward_data(
                alpha_BX_batch, eps=1e-7
            )
            b_ortho_batch = b_ortho_batch_bf16.to(torch.float32)

        # Save for backward
        ctx.save_for_backward(A_in, X_in)
        ctx.alpha = alpha
        ctx.beta = beta
        ctx.L = L
        ctx.ns_saved_tensors = ns_saved_tensors

        # Convert back to (B, D, L_padded, N)
        b_ortho = b_ortho_batch.transpose(1, 2)
        
        # Step 3: v_t = beta * v_{t-1} + b_t_ortho
        A_momentum = torch.full_like(A, beta, dtype=torch.float32)
        v_t = b_ortho.clone()
        PScan.pscan(A_momentum, v_t)  # v_t now contains the momentum states
        
        # Step 4: h_t = A_t * h_{t-1} + v_t
        h_t = v_t.clone()  # Start with v_t as the additive term
        PScan.pscan(A, h_t)  # h_t now contains the hidden states
        
        # Unpad and return
        return h_t.transpose(2, 1)[:, :L]
    
    @staticmethod
    def backward(ctx, grad_output_in):
        """
        Backward pass with recomputation for memory efficiency.
        
        Returns gradients for: A_in, X_in
        No gradients for alpha, beta (they are hyperparameters)
        """
        # Retrieve saved tensors
        A_in, X_in = ctx.saved_tensors
        ns_saved_tensors = ctx.ns_saved_tensors
        alpha = ctx.alpha
        beta = ctx.beta
        L = ctx.L
        L_padded = npo2(L)

        # --- RECOMPUTATION: Recreate forward pass ---
        A_in_fp32 = A_in.to(torch.float32)
        X_in_fp32 = X_in.to(torch.float32)

        if L == L_padded:
            A_padded = A_in_fp32.clone()
            X_padded = X_in_fp32.clone()
        else:
            A_padded = pad_npo2(A_in_fp32)
            X_padded = pad_npo2(X_in_fp32)
        
        A = A_padded.transpose(2, 1)      # (B, D, L_padded, N)
        BX = X_padded.transpose(2, 1)     # (B, D, L_padded, N)

        # Recompute b_t_ortho
        alpha_BX = alpha * BX
        alpha_BX_batch = alpha_BX.transpose(1, 2).contiguous()
        
        b_ortho_batch_bf16, _ = PScanMomentumNS5._newton_schulz_forward_data(
            alpha_BX_batch, eps=1e-7
        )
        b_ortho = b_ortho_batch_bf16.to(torch.float32).transpose(1, 2)

        # Recompute v_t
        A_momentum = torch.full_like(A, beta, dtype=torch.float32)
        v_t = b_ortho.clone()
        PScan.pscan(A_momentum, v_t)
        
        # Recompute h_t
        h_t = v_t.clone()
        PScan.pscan(A, h_t)

        # --- BACKPROPAGATION ---
        
        # Pad gradient
        if L == L_padded:
            grad_h = grad_output_in.clone().transpose(2, 1)
        else:
            grad_h = pad_npo2(grad_output_in).transpose(2, 1)

        # Backprop through h_t = A_t * h_{t-1} + v_t
        # grad_h is dL/dh_t, shape (B, D, L_padded, N)
        
        # Use reverse scan for gradient
        A_scan = torch.nn.functional.pad(A[:, :, 1:], (0, 0, 0, 1))
        grad_v = grad_h.clone()
        PScan.pscan_rev(A_scan, grad_v)  # grad_v is now dL/dv_t
        
        # Gradient w.r.t. A
        h_prev = torch.nn.functional.pad(h_t[:, :, :-1], (0, 0, 1, 0))
        grad_A = grad_v * h_prev
        
        # Backprop through v_t = beta * v_{t-1} + b_t_ortho
        A_momentum_scan = torch.nn.functional.pad(A_momentum[:, :, 1:], (0, 0, 0, 1))
        grad_b_ortho = grad_v.clone()
        PScan.pscan_rev(A_momentum_scan, grad_b_ortho)  # grad_b_ortho is dL/db_ortho
        
        # NOTE: We don't compute grad_beta since beta is a hyperparameter
        
        # Backprop through Newton-Schulz
        grad_b_ortho_batch = grad_b_ortho.transpose(1, 2).contiguous()
        
        if ns_saved_tensors is not None:
            grad_alpha_BX_batch = PScanMomentumNS5._newton_schulz_backward_data(
                grad_b_ortho_batch, ns_saved_tensors
            )
        else:
            # Fallback: use autograd (slower, but correct)
            alpha_BX_batch_req = alpha_BX_batch.detach().requires_grad_(True)
            with torch.enable_grad():
                b_ortho_recomp, _ = PScanMomentumNS5._newton_schulz_forward_data(
                    alpha_BX_batch_req, eps=1e-7
                )
                b_ortho_recomp = b_ortho_recomp.to(grad_b_ortho_batch.dtype)
            grad_alpha_BX_batch, = torch.autograd.grad(
                b_ortho_recomp, alpha_BX_batch_req, grad_b_ortho_batch
            )
        
        grad_alpha_BX = grad_alpha_BX_batch.transpose(1, 2)
        
        # Backprop through b_t = alpha * BX
        # NOTE: We don't compute grad_alpha since alpha is a hyperparameter
        grad_X = grad_alpha_BX * alpha
        
        # Unpad gradients
        grad_A = grad_A.transpose(2, 1)[:, :L]
        grad_X = grad_X.transpose(2, 1)[:, :L]

        # Return gradients only for A_in and X_in
        # None for alpha and beta (hyperparameters)
        return grad_A, grad_X, None, None

pscan_momentum_ns5 = PScanMomentumNS5.apply


class PScanMomentum(torch.autograd.Function):
    """
    Scans velocity and hidden states without Newton-Schulz.

    Forward signature:
        H = PScanMomentum.apply(A_in, X_in, alpha, beta)

    - A_in: (B, L, D, N)  -- multiplicative state matrix for h_t
    - X_in: (B, L, D, N)  -- "input contribution" BX_t (already deltaB * u_t)
    - alpha: scalar hyperparameter (scales the input contribution)
    - beta: scalar hyperparameter (momentum coefficient)

    Returns:
        H: (B, L, D, N) -- hidden states h_t
    """

    @staticmethod
    def forward(ctx, A_in, X_in, alpha, beta):
        L = X_in.size(1)
        L_padded = npo2(L)

        # cast to float32 for scan arithmetic
        A_fp32 = A_in.to(torch.float32)
        X_fp32 = X_in.to(torch.float32)

        if L == L_padded:
            A_padded = A_fp32.clone()
            X_padded = X_fp32.clone()
        else:
            A_padded = pad_npo2(A_fp32)
            X_padded = pad_npo2(X_fp32)

        # transpose to (B, D, L_padded, N) for in-place pscan primitives
        A = A_padded.transpose(2, 1)
        BX = X_padded.transpose(2, 1)

        # Step 1: momentum scan -> v_t = beta * v_{t-1} + alpha * BX_t
        alpha_BX = alpha * BX  # (B, D, L_padded, N)
        A_momentum = torch.full_like(A, beta, dtype=torch.float32)  # constant beta per-step

        v = alpha_BX.clone()
        PScan.pscan(A_momentum, v)  # v now contains v_t

        # Step 2: state scan -> h_t = A_t * h_{t-1} + v_t
        h = v.clone()
        PScan.pscan(A, h)  # h now contains h_t

        # save for backward recomputation
        ctx.save_for_backward(A_in, X_in)
        ctx.alpha = float(alpha)
        ctx.beta = float(beta)
        ctx.L = L

        return h.transpose(2, 1)[:, :L]

    @staticmethod
    def backward(ctx, grad_output_in):
        # retrieve saved inputs (original, unpadded)
        A_in, X_in = ctx.saved_tensors
        alpha = ctx.alpha
        beta = ctx.beta
        L = ctx.L
        L_padded = npo2(L)

        # recompute forward quantities (A, BX, v_t, h_t) for backward
        A_fp32 = A_in.to(torch.float32)
        X_fp32 = X_in.to(torch.float32)

        if L == L_padded:
            A_padded = A_fp32.clone()
            X_padded = X_fp32.clone()
        else:
            A_padded = pad_npo2(A_fp32)
            X_padded = pad_npo2(X_fp32)

        A = A_padded.transpose(2, 1)   # (B, D, L_padded, N)
        BX = X_padded.transpose(2, 1)  # (B, D, L_padded, N)

        # recompute momentum and state scans
        alpha_BX = alpha * BX
        A_momentum = torch.full_like(A, beta, dtype=torch.float32)

        v = alpha_BX.clone()
        PScan.pscan(A_momentum, v)  # v_t

        h = v.clone()
        PScan.pscan(A, h)  # h_t

        # prepare grad_h (dL/dh_t) padded and transposed -> (B, D, L_padded, N)
        if L == L_padded:
            grad_h = grad_output_in.clone().transpose(2, 1)
        else:
            grad_h = pad_npo2(grad_output_in).transpose(2, 1)

        # BACKPROP through h_t = A_t * h_{t-1} + v_t
        # Use reverse scan to produce grad_v (dL/dv_t) and accumulate grad_A from this equation.
        A_scan = torch.nn.functional.pad(A[:, :, 1:], (0, 0, 0, 1))  # left-shifted A with padding
        grad_v = grad_h.clone()
        PScan.pscan_rev(A_scan, grad_v)  # grad_v now dL/dv_t

        # grad_A from h-equation: grad_A = grad_v * h_prev
        h_prev = torch.nn.functional.pad(h[:, :, :-1], (0, 0, 1, 0))  # (B, D, L_padded, N)
        grad_A_from_h = grad_v * h_prev  # (B, D, L_padded, N)

        # BACKPROP through v_t = beta * v_{t-1} + alpha * BX_t
        # Use reverse scan on A_momentum to get gradients w.r.t. (alpha * BX)
        A_momentum_scan = torch.nn.functional.pad(A_momentum[:, :, 1:], (0, 0, 0, 1))
        grad_alpha_BX = grad_v.clone()
        PScan.pscan_rev(A_momentum_scan, grad_alpha_BX)  # grad_alpha_BX is dL/d(alpha * BX)

        # alpha is hyperparam -> no grad returned for it; gradient w.r.t BX = grad_alpha_BX * alpha
        grad_BX = grad_alpha_BX * alpha  # (B, D, L_padded, N)

        # grad_X is grad_BX (X_in was the BX passed in)
        grad_X = grad_BX

        # grad_A is grad_A_from_h (only from h equation)
        grad_A = grad_A_from_h

        # unpad and transpose grads back to (B, L, D, N)
        grad_A = grad_A.transpose(2, 1)[:, :L]
        grad_X = grad_X.transpose(2, 1)[:, :L]

        # return gradients for A_in and X_in, and None for alpha, beta
        return grad_A, grad_X, None, None

pscan_momentum = PScanMomentum.apply




# # Example usage:
# if __name__ == "__main__":
#     # Test the implementation
#     B, L, D, N = 2, 16, 64, 16
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     A = torch.randn(B, L, D, N, device=device, requires_grad=True)
#     X = torch.randn(B, L, D, N, device=device, requires_grad=True)
#     alpha = 0.02  # Learning rate scale (hyperparameter)
#     beta = 0.95   # Momentum coefficient (hyperparameter)
    
#     # Forward pass
#     H = pscan_momentum_ns5(A, X, alpha, beta)
    
#     # Backward pass
#     loss = H.sum()
#     loss.backward()
    
#     print(f"Output shape: {H.shape}")
#     print(f"A gradient shape: {A.grad.shape}")
#     print(f"X gradient shape: {X.grad.shape}")
#     print(f"Gradients computed successfully!")