from jax import numpy as jnp

from chemprop import args
# fingerprint.py in the same folder (modified chemprop code)
import fingerprint

from sklearn import metrics
import pandas as pd
import numpy as np

# molecular feature generation from chemprop
def inputGeneration(db, pt):
    evalArgs = [
        '--test_path', db,
        '--preds_path', './dummy.csv',
        '--checkpoint_path', pt,
        '--number_of_molecules', '2',
        '--smiles_columns', 'smi', 'smi'
    ]

    arg = args.FingerprintArgs().parse_args(evalArgs)
    x = np.array(fingerprint.molecule_fingerprint(arg))
    x = x.reshape(x.shape[0], x.shape[1])

    y = pd.read_csv(db)['vis'].values
    # 128-dimension
    return x[:, :128], y

class KernelRegression:
    def __init__(self, alpha = 0.1, gamma = 0.1):
        # l2 penalty
        self.alpha = alpha
        # gaussian parameter
        self.gamma = gamma

    # gaussian kernel
    def kernel(self, x, y):
        X = x[:, None, :]
        Y = y[None, :, :]
        D = ((X - Y) ** 2).sum(-1)
        return jnp.exp(-self.gamma * D)

    # fitting
    def fit(self, x, y):
        nSample, nFeature = x.shape

        self.xr = jnp.float32(x)
        K = self.kernel(self.xr, self.xr)
        self.W = y @ jnp.linalg.inv(self.alpha * jnp.eye(nSample) + K)
        
    # evaluation
    def eval(self, x):
        K = self.kernel(self.xr, x)
        return self.W @ K

# molecuular feature generation
xr, yr = inputGeneration(db = './db/public.csv', pt = './model.pt')

# fitting in log scale
yr = np.log10(yr)

# 300 training samples, 22 validation samples
xt, xv = xr[:300], xr[300:]
yt, yv = yr[:300], yr[300:]

# optimized value
# alpha 4.39455386e-04
# gamma 2.18623551e-05
model = KernelRegression(alpha = 4.39455386e-04, gamma = 2.18623551e-05)

# fitting
model.fit(xt, yt)
# evaluation
yp = model.eval(xr)

# correlation coefficients
# training set
print('Train: %.4f' %metrics.r2_score(yt, yp[:300]))
# validation set
print('Valid: %.4f' %metrics.r2_score(yv, yp[300:]))

# save results
df = pd.read_csv('./db/public.csv')
df['prd'] = 10 ** yp
df.to_csv('result.csv', index = False)