import numpy as np
import sys
np.random.seed(0)
P = np.random.normal(size=(4,4))

sys.stdout.write('{\n')
for r in P:
    sys.stdout.write('{')
    sys.stdout.write(str(r[0]))
    for e in r[1:]:
        sys.stdout.write(', ')
        sys.stdout.write(str(e))
    sys.stdout.write('},\n')
sys.stdout.write('}\n')
print '==='
print P
print np.linalg.inv(P)

