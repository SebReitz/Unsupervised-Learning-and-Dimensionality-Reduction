import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  confusion_matrix,adjusted_rand_score
from sklearn.cluster import KMeans,SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import homogeneity_score
from sklearn.decomposition import PCA,FastICA
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras import optimizers




'''Census Data'''
########################
'''Importing the dataset 1'''
data = pd.read_csv('/Users/sebastianreitz/Desktop/Georgia Tech/1. Semester/Machine Learning/Assignment 1/adult.csv')
data = data.drop(['educational-num','fnlwgt'],1)
data = data[data['workclass'] != '?']

# Check for NaNs: No NaNs in the Dataset
data = data.dropna()
# Simplify and drop capital loss
data['capital-gain'] = data['capital-gain']-data['capital-loss'] 
data = data.drop(['capital-loss'],1)
data_columns = data.columns.values.tolist()
data_columns = data_columns[:-1]
data_y = data.iloc[:,-1]

y = pd.get_dummies(data_y,drop_first=True)
X = pd.get_dummies(data[data_columns],drop_first=False)

#Scaling Data
sc = StandardScaler()
X = sc.fit_transform(X)
X = pd.DataFrame(X)



'''Visualizing Scaling'''
rnorm = np.random.randn
xx = rnorm(1000) * 10  
yy = np.concatenate([rnorm(500), rnorm(500) + 5])
fig, axes = plt.subplots(1, 3,figsize=(8,4))
axes[0].scatter(xx, yy)
axes[0].set_title('Random Gaussian Data')
km = KMeans(2)
clusters = km.fit_predict(np.array([xx, yy]).T)
axes[1].scatter(xx, yy, c=clusters, cmap='bwr')
axes[1].set_title('Non-normalised K-means')
clusters = km.fit_predict(np.array([xx/ 10, yy]).T)
axes[2].scatter(xx, yy, c=clusters, cmap='bwr')
axes[2].set_title('Normalised K-means')
# from:
#https://stats.stackexchange.com/questions/89809/is-it-important-to-scale-data-before-clustering



'''1.1 KMeans and WCSS'''
########################
wcss = []
for i in range(1,5):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) #Intertia is another name for the wcss, which can be drawn from the KMeans class

plt.figure()
plt.plot(range(1,5),wcss)
plt.title('Scaled WCSS by Cluster (Census Data)')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.show()

# Applying k-means to the dataset
kmeans = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10, random_state=0)
y_kmeans = kmeans.fit_predict(X)
y = y.iloc[:,0].values
adj_rand_index = adjusted_rand_score(y_kmeans,y)
print(adj_rand_index)


'''1.2 Expectation Maximization (GMM)'''
########################
gmm = GaussianMixture(n_components=2,n_init=10).fit(X)
labels = gmm.predict(X)
adj_rand_index = adjusted_rand_score(labels,y)
print(adj_rand_index)

hg_score = homogeneity_score(labels,y)
print(hg_score)


'''1.3 PCA'''
########################
pca = PCA(n_components = 2)
X1 = pca.fit_transform(X)

#X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum()) # gives explained variance of first n PCs

pca_optimal = PCA(n_components=0.95, svd_solver='full')
X2 = pca_optimal.fit_transform(X)

pca_o_var = pca_optimal.explained_variance_ratio_.cumsum()
pca_o_var2 = pca_optimal.explained_variance_ratio_
x_axis = np.linspace(0,len(pca_o_var),len(pca_o_var))

plt.figure()
plt.bar(x_axis, pca_o_var*100,label='PCs (Variance) Cumsum')
plt.bar(x_axis, pca_o_var2*100,label='PCs (Variance)',alpha=0.99)
plt.title('Unscaled Cumulative Sum of Principal Components covering 95% of Variance')
plt.xlabel('First {} PCs'.format(len(pca_o_var)))
plt.ylabel('Variance of Data in percent')
plt.legend()
plt.show()


'''Optimal PCA Cluster Value '''
pca_elements = np.linspace(25, 60, 26, endpoint=True)
#pca_elements = np.linspace(2, 40, 39, endpoint=True)
train_results = []
test_results = []
for e in pca_elements:
    print(e)
    data = pd.read_csv('/Users/sebastianreitz/Desktop/Georgia Tech/1. Semester/Machine Learning/Assignment 1/adult.csv')
    data = data.drop(['educational-num','fnlwgt'],1)
    data = data[data['workclass'] != '?']
    
    # Check for NaNs: No NaNs in the Dataset
    data = data.dropna()
    # Simplify and drop capital loss
    data['capital-gain'] = data['capital-gain']-data['capital-loss'] 
    data = data.drop(['capital-loss'],1)
    data_columns = data.columns.values.tolist()
    data_columns = data_columns[:-1]
    data_y = data.iloc[:,-1]
    
    y = pd.get_dummies(data_y,drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.25, random_state = 0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    '''PCA'''
#    pca = PCA(n_components = int(e))
#    X_train = pca.fit_transform(X_train)
#    X_test = pca.transform(X_test)
    '''ICA'''
    ica = FastICA(n_components=int(e))
    X_train = ica.fit_transform(X_train)
    X_test = ica.transform(X_test)   
    
    sgd = optimizers.SGD(lr=0.2)
    classifier = Sequential()
    classifier.add(Dense(int(len(X_train[0,:])/2),bias_initializer='random_normal', kernel_initializer = 'random_normal', activation = 'relu', input_dim = len(X_train[0,:])))  
    classifier.add(Dense(int(len(X_train[0,:])/2),bias_initializer='random_normal', kernel_initializer = 'random_normal', activation = 'relu')) 
    classifier.add(Dense(1,bias_initializer='random_normal', kernel_initializer = 'random_normal', activation = 'sigmoid')) 
    classifier.compile(optimizer=sgd, loss='binary_crossentropy',metrics=['accuracy'])
    classifier.fit(X_train,y_train,batch_size=100,nb_epoch=1)
    train_results.append(classifier.evaluate(X_train, y_train)[1]*100)
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5) 
    cm = confusion_matrix(y_test, y_pred)
    accuracy = (cm[0,0]+ cm[1,1])/len(X_test)
    test_results.append(accuracy*100)


plt.figure()
plt.plot(pca_elements, train_results,label='Train Results')
plt.plot(pca_elements, test_results,label='Test Results')
plt.axvline(pca_elements[np.argmax(test_results)],color='red',linestyle='--', alpha=0.3)
#plt.title('ANN: Number of Principal Components')
plt.title('ANN: Number of Independent Components')
plt.legend()
plt.ylabel('Accuracy in %')
#plt.xlabel('Number of Principal Components')
plt.xlabel('Number of Indipendent Components')
plt.show()


'''1.4 ICA'''
########################
ica = FastICA(n_components=26)
ica_transform = ica.fit_transform(X)


from scipy.stats import kurtosis
kurt = []
for i in range(1,40):
    X_transformed =FastICA(n_components=i,random_state=0).fit_transform(X)
    kurt.append(max(kurtosis(X_transformed,fisher=True)))

plt.figure()
plt.plot(range(1,40),kurt,label='Max Kurtosis')
plt.axvline(8,color='red',alpha=0.7,linestyle='--',label='Optimal # = 8')
plt.xlabel('Number of ICs')
plt.ylabel('Largest Kurtosis')
plt.legend()
plt.show()  


'''1.5 Random Projections'''
########################
from sklearn.random_projection import johnson_lindenstrauss_min_dim, GaussianRandomProjection
min_dimensions = []
eps = np.linspace(0.1,1,50,endpoint = False)
for i in eps:
    jldim = johnson_lindenstrauss_min_dim(len(X), eps=i)
    min_dimensions.append(jldim)
    
plt.figure()
plt.plot(eps,min_dimensions,label='D(Eta)')
plt.title('Minimum Dimensions J-L Lemma as Function of Eta')
plt.axhline(min_dimensions[-1],color='red',label='{} Dimensions'.format(
        min_dimensions[-1]),alpha=0.7,linestyle='--')
plt.xlabel('Eta')
plt.ylabel('Minimum Number of Dimensions')
plt.legend()
plt.show()


#X_rp = X.iloc[:200,:]
transformer = GaussianRandomProjection(eps=0.9999,n_components = 2, random_state=0)
X_rp = transformer.fit_transform(X)


'''1.6 Random Forest'''
########################
from sklearn.ensemble import ExtraTreesClassifier
forest = ExtraTreesClassifier(n_estimators=5000,
                              random_state=0)
forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
indices = indices[:20]

# Print the feature ranking
print("Feature ranking:")

for f in range(len(indices)):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature Importances")
plt.bar(range(len(indices)), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(len(indices)), indices)
plt.xlim([-1, len(indices)])
plt.xlabel('Features')
plt.ylabel('Relative Frequency of Most Feature Importance')
plt.show()

''' from 
http://scikit-learn.org/stable/auto_examples/
ensemble/plot_forest_importances.html
'''




'''PCA, KMeans/EM'''
########################
# Visualize the results on PCA-reduced data
reduced_data = PCA(n_components=2).fit_transform(X)

wcss_reduced = []
for i in range(1,20):
    kmeans_reduced = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state=0)
    kmeans_reduced.fit(reduced_data)
    wcss_reduced.append(kmeans_reduced.inertia_) #Intertia is another name for the wcss, which can be drawn from the KMeans class

plt.figure()
plt.plot(range(1,20),wcss_reduced)
plt.title('WCSS by Cluster pn PC with 2 Dimensions')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.show()


'''KMeans'''
kmeans = KMeans(init='k-means++', n_clusters=7, n_init=10)
kmeans.fit(reduced_data)

'''Expectation Maximization'''
import scipy.stats
gmm = GaussianMixture(n_components=7,n_init=10).fit(reduced_data)
centers = np.empty(shape=(gmm.n_components, reduced_data.shape[1]))
for i in range(gmm.n_components):
    density = scipy.stats.multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(reduced_data)
    centers[i, :] = reduced_data[np.argmax(density)]

h = 0.02    
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
#Z = gmm.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)

centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
#plt.scatter(centers[:, 0], centers[:, 1],
 #           marker='x', s=169, linewidths=3,
  #          color='w', zorder=10)
plt.title('KMC on PCA reduced Datasets')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()





''' PCA, KMeans and ANN'''
########################
Z = kmeans.predict(reduced_data)

ul_ann = []
ul_ann2 = []
for i in range(len(Z)):
    if Z[i] == 0:
        ul_ann.append(centroids[0,0])
        ul_ann2.append(centroids[0,1])
    if Z[i] == 1:
        ul_ann.append(centroids[1,0])
        ul_ann2.append(centroids[1,1])
    if Z[i] == 2:
        ul_ann.append(centroids[2,0])
        ul_ann2.append(centroids[2,1])
    if Z[i] == 3:
        ul_ann.append(centroids[3,0])
        ul_ann2.append(centroids[3,1])
    if Z[i] == 4:
        ul_ann.append(centroids[4,0])
        ul_ann2.append(centroids[4,1])
    if Z[i] == 5:
        ul_ann.append(centroids[5,0])
        ul_ann2.append(centroids[5,1])
    if Z[i] == 6:
        ul_ann.append(centroids[6,0])
        ul_ann2.append(centroids[6,1])


ul_ann = pd.DataFrame(ul_ann)
ul_ann['1'] = ul_ann2

sc = StandardScaler()
X = sc.fit_transform(ul_ann)

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size = 0.25, random_state = 0)

train_results = []
test_results = []

sgd = optimizers.SGD(lr=0.1)
classifier = Sequential()
classifier.add(Dense(len(X[0,:])*2, bias_initializer='random_normal', kernel_initializer = 'random_normal', activation = 'relu', input_dim = len(X[0,:])))
classifier.add(Dense(int(len(X[0,:])/2), bias_initializer='random_normal',kernel_initializer = 'random_normal', activation = 'relu')) 
classifier.add(Dense(1, bias_initializer='random_normal',kernel_initializer = 'random_normal', activation = 'sigmoid')) 
classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(X_train,y_train,batch_size=50,nb_epoch=10)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) 
cm = confusion_matrix(y_test, y_pred)
accuracy = (cm[0,0]+ cm[1,1])/len(X_test)
test_results=accuracy*100
train_results=classifier.evaluate(X_train, y_train)[1]*100

print('ANN Accuracy:')
print(accuracy)





###################################################################
'''Second Dataset'''
###################################################################



'''Importing the dataset 2'''
########################
data = pd.read_csv('/Users/sebastianreitz/Desktop/Georgia Tech/1. Semester/Machine Learning/Assignment 1/MLNasaDataset.csv')

X = data.iloc[:,:9].values
y = data.iloc[:,-1].values


''' Encoding Dependent Variable y'''
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)



sc = StandardScaler()
X1 = sc.fit_transform(X)
X1 = pd.DataFrame(X1)


'''2.1 KMeans and WCSS'''
########################
kmeans = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10, random_state=0)
y_kmeans = kmeans.fit_predict(X1)
adj_rand_index2 = adjusted_rand_score(y_kmeans,y)
print(adj_rand_index2)


wcss = []
for i in range(1,30):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) #Intertia is another name for the wcss, which can be drawn from the KMeans class

plt.figure()
plt.plot(range(1,30),wcss)
plt.title('WCSS by Cluster (Shuttle Data)')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.show()


kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state=0)
y_kmeans = kmeans.fit_predict(X)


wcss1 = []
for i in range(1,30):
    kmeans1 = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state=0)
    kmeans1.fit(X1)
    wcss1.append(kmeans1.inertia_) #Intertia is another name for the wcss, which can be drawn from the KMeans class

plt.figure()
plt.plot(range(1,30),wcss1)
plt.title('Scaled WCSS by Cluster (Shuttle Data)')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.show()


'''2.2 Expectation Maximization'''
########################
gmm = GaussianMixture(n_components=2,n_init=10).fit(X1)
labels = gmm.predict(X1)
adj_rand_index = adjusted_rand_score(labels,y)
print(adj_rand_index)

hg_score = homogeneity_score(labels,y)
print(hg_score)


'''2.3 PCA'''
########################
pca = PCA(n_components = 3)
X3 = pca.fit_transform(X1)
#X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum()) # gives explained variance of first n PCs

pca_optimal = PCA(n_components=0.95, svd_solver='full')
X2 = pca_optimal.fit_transform(X1)
# gives explained variance of first n PCs
pca_o_var = pca_optimal.explained_variance_ratio_.cumsum()
pca_o_var2 = pca_optimal.explained_variance_ratio_
x_axis = np.linspace(0,len(pca_o_var)-1,len(pca_o_var))

plt.figure()
plt.bar(x_axis, pca_o_var*100,label='PCs (Variance) Cumsum')
plt.bar(x_axis, pca_o_var2*100,label='PCs (Variance)',alpha=0.6)
plt.title('Cumulative Sum of Principal Components covering 95% of Variance')
plt.xlabel('First {} PCs'.format(len(pca_o_var)))
plt.ylabel('Variance of Data in percent')
plt.ylim(0,120)
plt.legend()
plt.show()


'''2.4 ICA'''
########################
ica = FastICA(n_components=9)
ica_transform = ica.fit_transform(X1)

kurt = []
for i in range(1,9):
    X_transformed =FastICA(n_components=i,random_state=0).fit_transform(X1)
    kurt.append(max(kurtosis(X_transformed,fisher=True)))

plt.figure()
plt.plot(range(1,9),kurt,label='Max Kurtosis')
plt.axvline(8,color='red',alpha=0.7,linestyle='--',label='Optimal # = 8')
plt.xlabel('Number of ICs')
plt.ylabel('Largest Kurtosis')
plt.legend()
plt.show()  


'''2.5 Random Projections'''
########################
from sklearn.random_projection import johnson_lindenstrauss_min_dim
jldim = johnson_lindenstrauss_min_dim(len(X), eps=0.5)

min_dimensions = []
eps = np.linspace(0.1,1,50,endpoint = False)
for i in eps:
    jldim = johnson_lindenstrauss_min_dim(len(X), eps=i)
    min_dimensions.append(jldim)
    
plt.figure()
plt.plot(eps,min_dimensions,label='D(Eta)')
plt.title('Minimum Dimensions J-L Lemma as Function of Eta')
plt.axhline(min_dimensions[-1],color='red',label='{} Dimensions'.format(
        min_dimensions[-1]),alpha=0.7,linestyle='--')
plt.xlabel('Eta')
plt.ylabel('Minimum Number of Dimensions')
plt.legend()
plt.show()

#X_rp = X.iloc[:200,:]
transformer = GaussianRandomProjection(eps=0.9999,n_components = 2, random_state=0)
X_rp = transformer.fit_transform(X1)


'''2.6 Random Forest'''
########################
from sklearn.ensemble import ExtraTreesClassifier
forest = ExtraTreesClassifier(n_estimators=5000,
                              random_state=0)
forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
indices = indices[:20]

# Print the feature ranking
print("Feature ranking:")

for f in range(len(indices)):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature Importances")
plt.bar(range(len(indices)), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(len(indices)), indices)
plt.xlim([-1, len(indices)])
plt.xlabel('Features')
plt.ylabel('Relative Frequency of Most Feature Importance')
plt.show()


