from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from  cvxopt import matrix , solvers
from sklearn.svm import SVC


np.random.seed(42)

means=[[2,2],[4,2]]
cov=[[.3,.2] ,[.2,.3]]
N=10

X0=np.random.multivariate_normal( means[0],cov,N ) # class 1
X1=np.random.multivariate_normal( means[1],cov,N ) # class -1
print(X0)
print(X1)
X=np.concatenate((X0.T ,X1.T) , axis=1)
X
y=np.concatenate((np.ones((1,N)) , -1*np.ones((1,N))) ,axis=1)
y

V= np.concatenate((X0.T ,-X1.T), axis=1)
K=matrix(V.T.dot(V)) # see definition of V, K near eq (8)

p=matrix(-np.ones((2*N,1))) # all one vector
#build G
G=matrix(-np.eye(2*N))
h=matrix(np.zeros((2*N,1)))
A=matrix(y) # the euqality constraint is actually y^T lambda=0
b=matrix(np.zeros((1,1)))

solvers.options['show_progress']=False
sol=  solvers.qp( K,p,G,h,A,b)
# sol = solvers.qp(K, p, G, h, A, b)
l=np.array(sol['x'])
print('Lambda = ')
print(l.T)



epsilon = 1e-6 # just a small number, greater than 1e-9
S = np.where(l > epsilon)[0]

VS = V[:, S]
XS = X[:, S]
yS = y[:, S]
lS = l[S]
# calculate w and b
w = VS.dot(lS)
b = np.mean(yS.T - w.T.dot(XS))

print('w = ', w.T)
print('b = ', b)


y1 = y.reshape((2*N,))
X1 = X.T # each sample is one row
clf = SVC(kernel = 'linear', C = 1e5) # just a big number

clf.fit(X1, y1)

w = clf.coef_
b = clf.intercept_
print('w = ', w)
print('b = ', b)