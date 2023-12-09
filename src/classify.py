import utils
import numpy as np
import test2
#import test_classify
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def PCA(X,n_retain):
    '''
    Argument: 
        X: X.shape=(n_sample,n_feature);
        n_retain: retain the first n_retain component of PCA,
    Returns: X1,eigvector,eigval=PCA(X,n_retain),
        X1: the coordinates of sample on PCA directions, X1.shape=(n_sample,n_retain)
        eigvector: first n_retain normalized PCA directions, eigvector.shape=(n_feature,n_retain)
        eigval: first n_retain eigenvalues of covariant matrix, listed in decreasing order, eigval.shape=(n_retain,)
        
    '''
    n_sample=X.shape[0]
    small_noise=0.01;
    X=X+small_noise*np.random.randn(X.shape[0],X.shape[1])
    X_mean=np.mean(X,axis=0);
    X=X-X_mean;
    sigma=(1/n_sample)*((X.T) @ X); #(n_sample,n_sample)
    eigvalue,eigvec=np.linalg.eigh(sigma)

    X1=(X @ eigvec);X1=X1[:,::-1]
    eigvec=eigvec[:,::-1]

    return (X1[:,0:n_retain],eigvec[:,0:n_retain],eigvalue[::-1])

def get_data(N_company=28,N_esg=50,time_start='2010-01-01',time_end='2023-10-27',time_interval=365):
    '''
    Argument: 
        N_company: Number of company we select. The default is 28. This is some preblem in data-fetching when N>28. I don't know why.
        N_esg: how many types of esg_score we want to use. This is some preblem in data-fetching when N>50. I don't know why either.
        time_start: There is no ESG data for most companies before 2010
        time_end:
        time_interval: ESG data is yearly
    
    Return:
        this function returns nothing. Instead it create two 2Darrray: esg_data, and price_data, 
        and they are saved in the files respectively as: 'esg_data.txt', and 'price_data.txt'.
        esg_data: esg_data.shape=(N_sample,N_esg). N_sample is roughly N_company*N_T, where N_T=(times_end-time_start)/time_interval
        price_data: price_data.shape=(N_sample,3). 
            Its first column only takes value in 1 or -1, indicating price increase or decrease.
            Its second column is the relative increase rate of price.
            Its third column is the price itself.
    '''

    # N_company=28; # N_company<=100, but best set 28
    company_list=utils.get_company_list(); #(500,)
    # company_list=company_list[0:N_company]; #(N_company,)
    # N_esg=50; # N_esg <= 82, but best set 50
    esg_list=utils.get_esg_list(); #(82,)

    

    esg_reading_list=[]
    for i in range(N_esg):
        esg_reading_list.append(('esg',esg_list[i]));

    company=utils.company(company_list[0])
    company.update_time_range(time_start,time_end,time_interval);
    temp_esg=(company.read_data(esg_reading_list)[1]).T;
    esg_data=temp_esg[1:(temp_esg.shape[0]),:];
    temp_price=company.read_single_data(('price','Close'));
    temp_increase_price=(temp_price[1:len(temp_price)]-temp_price[0:(len(temp_price)-1)])/temp_price[0:(len(temp_price)-1)];
    temp_sign_price=np.sign(temp_increase_price);temp_sign_price[temp_sign_price==0]=1;
    price_data=np.hstack((temp_sign_price[:,None],temp_increase_price[:,None],temp_price[1:len(temp_price),None]));
    print(esg_data.shape,price_data.shape)

    for i in range(1,N_company):
        company=utils.company(company_list[i])
        company.update_time_range(time_start,time_end,time_interval);
        temp_esg=(company.read_data(esg_reading_list)[1]).T;
        temp_esg1=temp_esg[1:(temp_esg.shape[0]),:];
        temp_price=company.read_single_data(('price','Close'));
        temp_increase_price=(temp_price[1:len(temp_price)]-temp_price[0:(len(temp_price)-1)])/temp_price[0:(len(temp_price)-1)];
        temp_sign_price=np.sign(temp_increase_price);temp_sign_price[temp_sign_price==0]=1;
        temp_price1=np.hstack((temp_sign_price[:,None],temp_increase_price[:,None],temp_price[1:len(temp_price),None]));
        
        esg_data=np.vstack((esg_data,temp_esg1))
        price_data=np.vstack((price_data,temp_price1))
        print(i,esg_data.shape,price_data.shape)
        # print(temp)
    esg_data[np.isnan(esg_data)==True]=0;

    with open('esg_data.txt','w') as f1 :
        np.savetxt(f1,esg_data)
    with open('price_data.txt','w') as f2 :
        np.savetxt(f2,price_data)

# # # -------------------------------------------------------------------------------------------------

### generate the data and read them
get_data();
esg_list=utils.get_esg_list();
esg_data= np.loadtxt('esg_data.txt') # (n_sample,n_feature)=(261,50)
price_data= np.loadtxt('price_data.txt') #(n_sample,3)=(261,3)

### PCA, print the eigvalue distribution
N_pca=50;
total_data,eigvec,eigvalue=PCA(esg_data,N_pca)

plt.figure();
plt.bar(np.arange(esg_data.shape[1]),eigvalue);
plt.xlabel('PCA axis')
plt.ylabel('Variance')
plt.title('Eigenvalue of covariant matrix');
plt.savefig('plt_eigvalue_of_PCA.jpg')

### PCA, print the first five eigenvector distribution
plt.figure();
for i in range(5):
    plt.plot(np.arange(esg_data.shape[1]),np.abs(eigvec[:,i])**2,'-o',markersize=3,label='PCA # {}'.format(i+1))
plt.legend();
plt.xlabel('features')
plt.ylabel('|eigvector|^2')
plt.title('Distribution of PCA eigvectors')
plt.savefig('plt_PCA_eigvector.png')

### PCA, print the leading components, this is the TABLE in the report.
for i in range(5):
    indexlst=np.argsort(eigvec[:,i]**2);indexlst=indexlst[::-1];
    for j in range(3):
        print('PCA#: {},ESG_name:{}, Strength:{}'.format(i+1,esg_list[indexlst[j]],eigvec[indexlst[j],i]**2))

### save the intermediate data after PCA, for later use
N_pca=50;
total_data,eigvec,eigvalue=PCA(esg_data,N_pca)
with open('esg_data_after_total_PCA.txt','w') as f3 :
    np.savetxt(f3,total_data)
total_data= np.loadtxt('esg_data_after_total_PCA.txt') # (n_sample,n_feature)=(261,50)

# # # --------------------------------------------------------------------------------------------------------

### draw the fancy 2D-diagram in the report, 
test2.Classify_all_methods([(total_data[:,[0,1]],price_data[:,0])],'plt_multiple_meethods_class_only_two_PCA2.jpg')

Classifier_list=[DecisionTreeClassifier(max_depth=5, random_state=42),
                 MLPClassifier(alpha=0.1, max_iter=1000, random_state=42),
                 AdaBoostClassifier(random_state=42),
                 QuadraticDiscriminantAnalysis()];
Classifier_name_list=['Decision Tree',
                      'Neural Networks',
                      'AdaBoost',
                      'QDA']

### study how the classification accuracy changes wrt number of PCA retained
test_accuracy=np.zeros((4,50));
train_accuracy=np.zeros((4,50));
for N_pca in range(1,51):
    X=total_data[:,range(0,N_pca)];y=price_data[:,0];
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42);
    for iclass in range(4):
        clf = Classifier_list[iclass].fit(X_train, y_train)
        test_accuracy[iclass,N_pca-1]=clf.score(X_test,y_test)
        train_accuracy[iclass,N_pca-1]=clf.score(X_train,y_train)

for iclass in range(4):
    plt.figure()
    plt.plot(range(1,51),test_accuracy[iclass,:],'r-o',label='test')
    plt.plot(range(1,51),train_accuracy[iclass,:],'g-o',label='train')
    plt.legend()
    plt.xlabel('Number of retained PCAs')
    plt.ylabel('Predicting Accuracy')
    plt.title(Classifier_name_list[iclass])
    plt.savefig('plt_accuracy_wrt_Npca_{}.jpg'.format(Classifier_name_list[iclass]))
    plt.show();

# # # ------------------------------------------------------------------------------------------------

### Study how NN is influenced by regularization
N_pca=28;
X=total_data[:,range(0,N_pca)];y=price_data[:,0];
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42);

regularize_list=np.array([0.01,0.05,0.1,0.5,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20]);
test_accuracy=np.zeros(len(regularize_list));
train_accuracy=np.zeros(len(regularize_list));

for ireg in range(len(regularize_list)):
    clf=MLPClassifier(alpha=regularize_list[ireg], max_iter=2000, random_state=42).fit(X_train,y_train);
    test_accuracy[ireg]=clf.score(X_test,y_test);
    train_accuracy[ireg]=clf.score(X_train,y_train);

plt.figure()
plt.plot(regularize_list,test_accuracy,'r-o',markersize=3,label='test')
plt.plot(regularize_list,train_accuracy,'g-o',markersize=3,label='train')
plt.legend()
plt.xlabel('Regularization strenth')
plt.ylabel('Predicting Accuracy')
plt.title('Effect of regularization')
plt.savefig('plt_regularization_in_MLP.jpg')
plt.show();

### Study how NN is influenced by width of NN
N_pca=28;
X=total_data[:,range(0,N_pca)];y=price_data[:,0];
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42);

depth_list=np.arange(10,200,10);
test_accuracy=np.zeros(len(depth_list));
train_accuracy=np.zeros(len(depth_list));

for i in range(len(depth_list)):
    clf=MLPClassifier(alpha=10,hidden_layer_sizes=(depth_list[i],), max_iter=2000, random_state=42).fit(X_train,y_train);
    test_accuracy[i]=clf.score(X_test,y_test);
    train_accuracy[i]=clf.score(X_train,y_train);

plt.figure()
plt.plot(depth_list,test_accuracy,'r-o',label='test')
plt.plot(depth_list,train_accuracy,'g-o',label='train')
plt.legend()
plt.xlabel('Number of Neurons in each layer')
plt.ylabel('Predicting Accuracy')
plt.title('Effect of Width of NN')
plt.savefig('plt_Width_in_MLP.jpg')
plt.show();

### Study how Decision tree is influenced by depth of tree
N_pca=28;
X=total_data[:,range(0,N_pca)];y=price_data[:,0];
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42);

depth_list=np.arange(2,15);
test_accuracy=np.zeros(len(depth_list));
train_accuracy=np.zeros(len(depth_list));

for i in range(len(depth_list)):
    clf=DecisionTreeClassifier(max_depth=depth_list[i], random_state=42).fit(X_train,y_train);
    test_accuracy[i]=clf.score(X_test,y_test);
    train_accuracy[i]=clf.score(X_train,y_train);

plt.figure()
plt.plot(depth_list,test_accuracy,'r-o',label='test')
plt.plot(depth_list,train_accuracy,'g-o',label='train')
plt.legend()
plt.xlabel('Maximal depth of tree')
plt.ylabel('Predicting Accuracy')
plt.title('Effect of Maximal depth in Decision Tree')
plt.savefig('plt_depth_in_tree.jpg')
plt.show();

### Study how decision tree is influenced by number of samples in ending leaves
N_pca=28;
X=total_data[:,range(0,N_pca)];y=price_data[:,0];
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42);

depth_list=np.arange(2,20);
test_accuracy=np.zeros(len(depth_list));
train_accuracy=np.zeros(len(depth_list));

for i in range(len(depth_list)):
    clf=DecisionTreeClassifier(min_samples_split=depth_list[i], random_state=42).fit(X_train,y_train);
    test_accuracy[i]=clf.score(X_test,y_test);
    train_accuracy[i]=clf.score(X_train,y_train);

plt.figure()
plt.plot(depth_list,test_accuracy,'r-o',label='test')
plt.plot(depth_list,train_accuracy,'g-o',label='train')
plt.legend()
plt.xlabel('Sample size at ending')
plt.ylabel('Predicting Accuracy')
plt.title('Effect of Sample size in the leaf of Decision Tree')
plt.savefig('plt_sample_size_in_tree.jpg')
plt.show();

### Study how Adaboost is influenced by number of estimators
N_pca=28;
X=total_data[:,range(0,N_pca)];y=price_data[:,0];
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42);

depth_list=np.arange(10,200,5);
test_accuracy=np.zeros(len(depth_list));
train_accuracy=np.zeros(len(depth_list));

for i in range(len(depth_list)):
    clf=AdaBoostClassifier(n_estimators=depth_list[i],random_state=42).fit(X_train,y_train);
    test_accuracy[i]=clf.score(X_test,y_test);
    train_accuracy[i]=clf.score(X_train,y_train);

plt.figure()
plt.plot(depth_list,test_accuracy,'r-o',label='test')
plt.plot(depth_list,train_accuracy,'g-o',label='train')
plt.legend()
plt.xlabel('Maximal number of estimators')
plt.ylabel('Predicting Accuracy')
plt.title('Effect of Maximal number of estimator in AdaBoost')
plt.savefig('plt_num_estimators_AdaBoost.jpg')
plt.show();

# # # --------------------------------------------------------------------------------------------