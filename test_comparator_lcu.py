"""Exact small-register validation of the folded comparator LCU block encoding."""
import numpy as np

# 1D Fourier index k = s*T + r. Uniform register j in [0,T), s is top bit.
# V_k diagonal with phase (-1)^{(j<r) XOR s};
# LCU postselected block B_k=<u|(I-V_k)/2|u>.

def lcu_amplitude(k,m):
    T=1<<(m-1)
    s,r=divmod(k,T)
    v=np.array([(-1.0)**(int(j<r)^s) for j in range(T)])
    return float(np.mean((1-v)/2.0))

def truth(k,m):
    M=1<<m
    return min(k,M-k)/(M/2)


def test():
    max_err=0.0;case_count=0
    for m in range(2,10):
        M=1<<m
        for k in range(M):
            delta=abs(lcu_amplitude(k,m)-truth(k,m))
            max_err=max(max_err,delta)
            case_count+=1
            assert delta <= 1e-14, (m,k,delta)
        assert lcu_amplitude(M//2,m)==1.0  # Nyquist
        assert lcu_amplitude(0,m)==0.0
    # Independent linear-algebra check: apply the selector-H / controlled(-V) / selector-H
    # circuit to |0>_sel |u>_j, then project onto the same state.
    H=np.array([[1.,1.],[1.,-1.]])/np.sqrt(2)
    for m in range(2,6):
        T=1<<(m-1)
        u=np.ones(T)/np.sqrt(T)
        u0=np.concatenate((u,np.zeros(T)))
        Hfull=np.kron(H,np.eye(T))
        for k in range(1<<m):
            s,r=divmod(k,T)
            signs=np.array([(-1.0)**(int(j<r)^s) for j in range(T)])
            controlled_neg_V=np.diag(np.r_[np.ones(T),-signs])
            result=Hfull@controlled_neg_V@Hfull@u0
            got=np.vdot(u,result[:T]).real
            assert abs(got-truth(k,m))<1e-13,(m,k,got,truth(k,m))
    print('PASS explicit selector/LCU matrices, m=2..5, including Nyquist and zero index.')
    print(f'PASS {case_count} frequencies, max absolute block-amplitude error={max_err:g}')
    print('Nyquist value one is implemented with s=1, r=0: no overflow in r comparator.')

if __name__=='__main__':test()
