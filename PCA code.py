import numpy as np
import pandas as pd
#the matrix they give you
#fill in your self
A = np.matrix([[1,2,3,4],
               [5,5,6,7],
               [1,4,2,3],
               [5,3,2,1],
               [8,1,2,2]])
df = pd.DataFrame(A,columns  = ['f1','f2','f3','f4'])
print("OUR DATAFTAME")
print("-------------------------")
print("-------------------------")
print(df)
df_std  = (df - df.mean()) / (df.std())
print("OUR STANDARDIZED DARAFRAME")
print(df_std)
print("-------------------------")
print("-------------------------")
#USE df_conv and conv_mat based on whether the question says this is a sample or the whole population
df_cov = np.cov(df_std.T, bias = 1)
print("Covariance population formula (divide by N)")
print(df_cov)
print("-------------------------")
print("-------------------------")
cov_mat = np.cov(df_std.T, bias = 0)
print("Covariance sample formula (divide by N-1)")
print(cov_mat)
print("-------------------------")
print("-------------------------")
eigen_val, eigen_vectors = np.linalg.eig(cov_mat)
print("OUR eigen_values")
print(eigen_val)
print("-------------------------")
print("-------------------------")
print("OUR eigen_vectors")
print(eigen_vectors)
print("-------------------------")
print("-------------------------")
#n_components is our k value
#change this based of how many components the question asks for
n_components=3
top_eigen_vectors = eigen_vectors[:,:n_components]
print("OUR TOP K EIGNE VECTORS")
print(top_eigen_vectors)
print("-------------------------")
print("-------------------------")
transformed_data = np.matmul(np.array(df_std),top_eigen_vectors)
principalCompDf = pd.DataFrame(data = transformed_data, columns = ['principal component '+ str(i+1) for i in range(n_components)])
print("OUR PCA RESULTS")
print(principalCompDf)
