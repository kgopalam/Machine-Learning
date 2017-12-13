
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
print(cancer.DESCR) # Print the data set description
#cancer.keys()
#cancer['feature_names']
cancerdf = pd.DataFrame(data = cancer.data,
				   columns =['mean radius', 'mean texture', 'mean perimeter', 'mean area','mean smoothness', 'mean compactness', 'mean concavity','mean concave points', 'mean symmetry', 'mean fractal dimension','radius error', 'texture error', 'perimeter error', 'area error','smoothness error', 'compactness error', 'concavity error','concave points error', 'symmetry error', 'fractal dimension error','worst radius', 'worst texture', 'worst perimeter', 'worst area','worst smoothness', 'worst compactness', 'worst concavity','worst concave points', 'worst symmetry', 'worst fractal dimension'],index = pd.RangeIndex(start=0, stop=569, step=1),dtype = float)
	
cancerdf['target'] = cancer['target']
cancerdf.target = cancerdf.target.astype(float)
ldata = cancerdf.groupby(['target']).size()
ldata.index = ['malignant','benign']
	
X = cancerdf.iloc[:,0:30]
y = cancerdf.iloc[:,30]
print(X)	
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
	
knn = KNeighborsClassifier(n_neighbors = 1) 
knn.fit(X_train, y_train)
	
Xm = cancerdf.mean()[:-1].values.reshape(1, -1)
cancer_prediction = knn.predict(Xm)
cancer_prediction_test=	knn.predict(X_test)
cancer_mean_accuracy = knn.score(X_test, y_test)
print(cancer_mean_accuracy)
	
# Find the training and testing accuracies by target value (i.e. malignant, benign)
mal_train_X = X_train[y_train==0]
mal_train_y = y_train[y_train==0]
ben_train_X = X_train[y_train==1]
ben_train_y = y_train[y_train==1]

mal_test_X = X_test[y_test==0]
mal_test_y = y_test[y_test==0]
ben_test_X = X_test[y_test==1]
ben_test_y = y_test[y_test==1]
		
scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y),knn.score(mal_test_X, mal_test_y),knn.score(ben_test_X, ben_test_y)]

print(scores)
plt.figure()
# Plot the scores as a bar chart
bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])
# directly label the score onto the bars
for bar in bars:
	height = bar.get_height()
	plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height,2),ha='center', color='w', fontsize=11)

# remove all the ticks (both axes), and tick labels on the Y axis
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

# remove the frame of the chart
for spine in plt.gca().spines.values():
	spine.set_visible(False)

plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest','Benign\nTest'],alpha=0.8);
plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)
plt.show()