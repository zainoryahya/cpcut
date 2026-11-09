"""Symbolic checks of the C2 rational quintic (Hussain et al. 2018): interpolation conditions,
reduction to the quintic Hermite polynomial, and Bernstein coefficients of the numerator N of S'."""
import sympy as sp
th,a,b,g,h=sp.symbols('theta alpha beta gamma h',positive=True)
fi,di,Di,fj,dj,Dj=sp.symbols('f_i d_i D_i f_j d_j D_j',real=True)   # j = i+1
# Hussain, Hussain & Yameen (2018), eq. (1)
A=[a*fi,
   (3*a+b)*fi + a*h*di,
   (3*a+3*b+g)*fi + (2*a+b)*h*di + sp.Rational(1,2)*a*h**2*Di,
   (3*g+3*b+a)*fj - (2*g+b)*h*dj + sp.Rational(1,2)*g*h**2*Dj,
   (3*g+b)*fj - g*h*dj,
   g*fj]
p=sum((1-th)**(5-k)*th**k*A[k] for k in range(6))
q=a*(1-th)**2+b*(1-th)*th+g*th**2
S=p/q
# verify interpolation conditions (2): S, S', S'' with x-derivative = (1/h) d/dtheta
chk=[sp.simplify(S.subs(th,0)-fi), sp.simplify(S.subs(th,1)-fj),
     sp.simplify(sp.diff(S,th).subs(th,0)/h-di), sp.simplify(sp.diff(S,th).subs(th,1)/h-dj),
     sp.simplify(sp.diff(S,th,2).subs(th,0)/h**2-Di), sp.simplify(sp.diff(S,th,2).subs(th,1)/h**2-Dj)]
print("interpolation checks (all 0?):",chk)
# reduces to quintic Hermite at (1,2,1)?
qh=sp.expand(sp.simplify(S.subs({a:1,b:2,g:1})))
print("alpha=1,beta=2,gamma=1 -> polynomial:", sp.Poly(sp.simplify(qh),th).degree())
# cut-off segment: flat end at theta=0
Dd,s,s2=sp.symbols('Delta s s2',real=True)       # Delta = f_j - f_i (drop), s = V'(delta), s2 = V''(delta)
sub={fi:0,di:0,Di:0,fj:Dd,dj:s,Dj:s2}            # shift f_i to 0 (monotonicity is shift-invariant)
N=sp.expand((sp.diff(p,th)*q-p*sp.diff(q,th)).subs(sub))
deg=sp.Poly(N,th).degree(); print("numerator of S' degree:",deg)
# Bernstein coefficients of N on [0,1]
B=[]
Pc=sp.Poly(N,th).all_coeffs()[::-1]   # power coefficients c_k
for k in range(deg+1):
    B.append(sp.factor(sum(sp.binomial(k,j)/sp.binomial(deg,j)*Pc[j] for j in range(k+1))))
for k,bk in enumerate(B): print(f"B{k} =",bk)

