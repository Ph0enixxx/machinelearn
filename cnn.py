import copy,numpy as np

np.random.seed(0)

sigmod = lambda x: 1/(1+np.exp(-x))
sigmod_d = lambda x:x*(1-x)

int2bin = {}
bin_dim = 8

max_num = pow(2,bin_dim)
binary = np.unpackbits(np.array([range(max_num)],dtype=np.uint8).T,axis=1)

print(binary)
for i in range(max_num):
    int2bin[i] = binary[i]

alpha = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1

syn_0 = 2*np.random.random((input_dim,hidden_dim)) - 1
syn_1 = 2*np.random.random((hidden_dim,output_dim)) - 1
syn_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1

syn_0_up = np.zeros_like(syn_0)
syn_1_up = np.zeros_like(syn_1)
syn_h_up = np.zeros_like(syn_h)


################

for j in range(10000):
    a_int = np.random.randint(max_num/2)
    a = int2bin[a_int]

    b_int = np.random.randint(max_num/2)
    b = int2bin[b_int]
    c_int = a_int + b_int
    c = int2binary[c_int]

    d = np.zeros_like(c)
    overallError = 0
    l_2_d = list()
    l_1_v = list()
    l_1_v.append(np.zeros(hidden_dim))
    for pos in range(bin_dim):
        X = np.array([[a[bin_dim - pos - 1],b[bin_dim - position - 1]]])
        y = np.array([[c[bin_dim - pos - 1]]]).T
        l_1 = sigmod(np.dot(X,syn_0) + np.dot(l_1_v[-1],syn_h))
        l_2 = sigmod(np.dot(l_1,syn_1)

        l_2_err = y - l_2
        l_2_d.append((l_2_err)*sigmod_d(l_2))
        overallError += np.abs(layer_2_error[0])
        d[bin_dim - pos - 1] = np.round(l_2[0][0])
        
    
