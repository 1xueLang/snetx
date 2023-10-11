#include "C/surrogate.in"

template<typename scalar_t> __global__ void Leaky_integrate(
    const scalar_t * x, 
    scalar_t * spike,
    const scalar_t tau, 
    const scalar_t v_th,
    const scalar_t v_reset,
    const int batch,
    const int step,
    const int dim
) {
    int idx_th = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx_th >= batch * dim) return;
    int posi = (idx_th - idx_th % dim) * step + (idx_th % dim);
    scalar_t potential = 0.0;

    for (int i = 0;i < step;++i, posi += dim) {
        potential = x[posi] + potential / tau;
        spike[posi] = potential >= v_th ? 1 : 0;
        potential = potential >= v_th ? v_reset : potential;
    }
}

template<typename scalar_t> __global__ void Leaky_integrate_FP(
    const scalar_t * x, 
    scalar_t * psps, 
    scalar_t * spike, 
    const scalar_t tau, 
    const scalar_t v_th, 
    const scalar_t v_reset, 
    const int batch, 
    const int step, 
    const int dim
) {
    int idx_th = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx_th >= batch * dim) return;
    int posi = (idx_th - idx_th % dim) * step + (idx_th % dim);
    scalar_t potential = 0.0;

    for (int i = 0;i < step;++i, posi += dim) {
        potential = x[posi] + potential / tau;
        spike[posi] = potential >= v_th ? 1 : 0;
        psps[posi] = potential;
        potential = potential >= v_th ? v_reset : potential;
    }

}

template<typename scalar_t> __global__ void Leaky_integrate_BP(
    const scalar_t * psps, 
    const scalar_t * grad_out, 
    scalar_t * grad_x,
    const scalar_t tau, 
    const scalar_t v_th, 
    const int batch, 
    const int step, 
    const int dim, 
    const int suro, 
    const scalar_t alpha
) {
    int idx_th = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx_th >= batch * dim) return;
    int posi = idx_th * step + (dim - idx_th % dim) * (step - 1);
    scalar_t du = 0.0, u = 0.0;

    for (int i = step - 1;i >= 0;--i, posi -= dim) {
        u = psps[posi];
        scalar_t over_th = u - v_th, sg = 0;
        
        __SWITCH_ON_SURO_FUNC__(suro)
        
        du = grad_out[posi] * sg + du * (1 - scalar_t(over_th >= 0) - u * sg) / tau;
        
        grad_x[posi] = du;
    }
}

template<typename scalar_t> __global__ void Leaky_integrate_detached_BP(
    const scalar_t * psps, 
    const scalar_t * grad_out, 
    scalar_t * grad_x,
    const scalar_t tau, 
    const scalar_t v_th, 
    const int batch, 
    const int step, 
    const int dim, 
    const int suro, 
    const scalar_t alpha
) {    
    int idx_th = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx_th >= batch * dim) return;
    int posi = idx_th * step + (dim - idx_th % dim) * (step - 1);
    scalar_t du = 0.0, u = 0.0;

    for (int i = step - 1;i >= 0;--i, posi -= dim) {
        u = psps[posi];
        scalar_t over_th = u - v_th, sg = 0;
        
        __SWITCH_ON_SURO_FUNC__(suro)
        
        du = over_th < 0 ? grad_out[posi] * sg + du / tau : grad_out[posi] * sg;
        
        grad_x[posi] = du;
    }
}
