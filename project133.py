import pandas as pd
from sklearn.cluster import KMeans
import csv
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("project_131.csv")
#print(df.head())
x = df.iloc[:,[4,5]].values
#print(x)
Wcss = []

for i in range(1,11):
    Kmeans = KMeans(n_clusters=i,init = 'k-means++',random_state=42)
    Kmeans.fit(x)
    Wcss.append(Kmeans.inertia_)
#print(Wcss)

plt.title("Elbow Method")
plt.xlabel("NO of Clusters")
plt.ylabel("WCSS")
plt.plot(range(1,11),Wcss)
#plt.show()
Kmeans = KMeans(n_clusters = 3,init = 'k-means++',random_state = 42)
y_Kmeans = Kmeans.fit_predict(x)
print(y_Kmeans)
sns.scatterplot(x[y_Kmeans == 0,0],x[y_Kmeans==0,1],color = 'yellow',label = 'cluster1')
sns.scatterplot(x[y_Kmeans == 1,0],x[y_Kmeans==1,1],color = 'blue',label = 'cluster2')
sns.scatterplot(x[y_Kmeans == 2,0],x[y_Kmeans==2,1],color = 'green',label = 'cluster3')
sns.scatterplot(Kmeans.cluster_centers_[:,0],Kmeans.cluster_centers_[:,1],color = 'red',label='centroid',s = 100,marker = ',')
plt.grid(False)
plt.title("Clusters of Mass and Radius")
plt.xlabel("Radius")
plt.ylabel("Mass")
plt.show()