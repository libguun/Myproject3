import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import datetime


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift, estimate_bandwidth
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#RFM func
def rfm(mdata, date_col, order_id_col, total_col, st, form):
    st.dataframe(mdata)
    st.write(mdata.shape)
    mdata['Order Date'] = pd.to_datetime(mdata['Order Date'], format='%Y-%m-%d')

    df_RFM = mdata.groupby('Customer ID').agg({date_col: lambda y: (mdata[date_col].max().date() - y.max().date()).days,
                                        order_id_col: lambda y: len(y.unique()),  
                                        total_col: lambda y: round(y.sum(),2)})
    if(len(df_RFM.columns)>2):
        df_RFM.columns = ['Recency', 'Frequency', 'Monetary']
        df_RFM = df_RFM.sort_values('Monetary', ascending=False)
        quantiles = df_RFM.quantile(q=[0.5])
        df_RFM['R']=np.where(df_RFM['Recency']<=int(quantiles.Recency.values), 2, 1)
        df_RFM['F']=np.where(df_RFM['Frequency']>=int(quantiles.Frequency.values), 2, 1)
        df_RFM['M']=np.where(df_RFM['Monetary']>=int(quantiles.Monetary.values), 2, 1)

        df_RFM.loc[(df_RFM['R']==2) & (df_RFM['M']==2),'class'] = 1
        df_RFM.loc[(df_RFM['R']==1) & (df_RFM['M']==2),'class'] = 2
        df_RFM.loc[(df_RFM['R']==2) & (df_RFM['M']==1),'class'] = 3
        df_RFM.loc[(df_RFM['R']==1) & (df_RFM['M']==1),'class'] = 4
        df_RFM['class']=df_RFM['class'].astype(int)

        result = pd.merge(mdata, df_RFM, on="Customer ID")

        train_data=result[['Customer ID','Recency','Frequency','Monetary','class']].copy()

        customers = train_data['Customer ID'].unique()
        customer_df=pd.DataFrame(customers, columns=['Customer ID'])

        train_data_merged = pd.merge(left=customer_df, right=train_data, left_on='Customer ID', right_on='Customer ID')

        train_data_merged=train_data_merged.drop_duplicates()
        train_data_merged = train_data_merged.reset_index(drop=True)
        st.dataframe(train_data_merged)
        st.write(train_data_merged.shape)
        return train_data_merged
    else:
        st.info('baganuudaa songono uu')

        

#Decision Tree
def dTree(x_train, y_train, x_test, y_test, st):
    model = DecisionTreeClassifier() 
    model.fit(x_train, y_train)

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    dt_train_score = model.score(x_train, y_train)
    dt_test_score = model.score(x_test, y_test)

    st.write("**Гүйцэтгэл :**",round(dt_test_score*100,2),"%")

#Random Forest
def rForest(x_train, y_train, x_test, y_test, st):
    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    rf_train_score = model.score(x_train, y_train)
    rf_test_score = model.score(x_test, y_test)

    st.write("**Гүйцэтгэл :**",round(rf_test_score*100,2),"%")


#Logistic Regression
def lRegression(x_train, y_train, x_test, y_test, st):
    model = LogisticRegression()
    model.fit(x_train, y_train)

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    lr_train_score = model.score(x_train, y_train)
    lr_test_score = model.score(x_test, y_test)

    st.write("**Гүйцэтгэл :**",round(lr_test_score*100,2),"%")

#Support Vector Machine
def SVM(x_train, y_train, x_test, y_test, st):
    model = SVC()
    model.fit(x_train, y_train)

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    svm_train_score = model.score(x_train, y_train)
    svm_test_score = model.score(x_test, y_test)

    st.write("**Гүйцэтгэл :**",round(svm_test_score*100,2),"%")

#Naive Bayes
def nBayes(x_train, y_train, x_test, y_test, st):
    model = GaussianNB()
    model.fit(x_train, y_train)

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    nb_train_score = model.score(x_train, y_train)
    nb_test_score = model.score(x_test, y_test)

    st.write("**Гүйцэтгэл** :",round(nb_test_score*100,2),"%")

#Kmeans
def kMeans(X, k, df, st):
    scaler = StandardScaler()
    scaler.fit(X)
    selected_data_std=scaler.transform(X)
    selected_data_std_df=pd.DataFrame(selected_data_std, columns=['Recency','Frequency','Monetary'])
    matrix = selected_data_std_df.to_numpy()

    kmeans = KMeans(n_clusters=k, random_state=0).fit(selected_data_std_df)
    plt.hist(kmeans.labels_)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    cluster_df = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=selected_data_std_df.columns)

    cluster_df_std = pd.DataFrame(kmeans.cluster_centers_, columns=selected_data_std_df.columns)
    
    selected_data_with_X = df.iloc[:,1:4] 
    selected_data_with_X['cluster'] = kmeans.labels_
    for i in range(k):
        selected_data_with_X.loc[selected_data_with_X['cluster'] == i].value_counts(normalize=True)

    pca = PCA(n_components=3)
    pca.fit(selected_data_std)
    pca.components_
    X1 = pd.DataFrame(pca.components_, columns =['Recency','Frequency','Monetary'])

    st.write('Selected data has', len(X1.columns), 'features')
    st.write('3 principal components has', np.sum(pca.explained_variance_ratio_), 'total variance explanation')

    selected_data_std_pca = pca.transform(selected_data_std)
    selected_data_std_pca = pd.DataFrame(selected_data_std_pca)
    selected_data_std_pca
    
    plt.scatter(selected_data_std_pca[0], selected_data_std_pca[1], c=kmeans.labels_, cmap='Paired', alpha=0.8)
    st.pyplot()

    plt.scatter(selected_data_std_pca[0], selected_data_std_pca[2], c=kmeans.labels_, cmap='Paired', alpha=0.8)
    st.pyplot()

    plt.scatter(selected_data_std_pca[1], selected_data_std_pca[2], c=kmeans.labels_, cmap='Paired', alpha=0.8)
    st.pyplot()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.scatter(selected_data_std_pca[0], selected_data_std_pca[1], selected_data_std_pca[2], c=kmeans.labels_, cmap='Paired')
    st.pyplot()

#Mean-shift
def meanShift(X, st):
    scaler = StandardScaler()
    scaler.fit(X)
    selected_data_std=scaler.transform(X)
    selected_data_std_df=pd.DataFrame(selected_data_std, columns=['Recency','Frequency','Monetary'])
    bandwidth = estimate_bandwidth(selected_data_std, quantile=0.2, n_samples=500)
    meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    meanshift.fit(selected_data_std)
    labels = meanshift.labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    st.write('Estimated number of clusters: ' + str(n_clusters_))
    y_pred  = meanshift.predict(selected_data_std)
    plt.hist(meanshift.labels_)
    st.pyplot()

    cluster_df_std = pd.DataFrame(meanshift.cluster_centers_, columns=selected_data_std_df.columns)

    cluster_df = pd.DataFrame(scaler.inverse_transform(meanshift.cluster_centers_), columns=selected_data_std_df.columns)
    selected_data_std_df['cluster'] = meanshift.labels_
    pd.Series(meanshift.labels_).value_counts(normalize=True)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca.fit(selected_data_std)
    pca.components_
    X1 = pd.DataFrame(pca.components_, columns =['Recency','Frequency','Monetary'])
    print('Selected data has', len(X1.columns), 'features')
    print('3 principal components has', np.sum(pca.explained_variance_ratio_), 'total variance explanation')

    selected_data_std_pca = pca.transform(selected_data_std)
    selected_data_std_pca = pd.DataFrame(selected_data_std_pca, columns = ['pca1', 'pca2', 'pca3'])

    plt.scatter(selected_data_std_pca['pca1'], selected_data_std_pca['pca2'], c=meanshift.labels_, cmap='Paired', alpha=0.8)
    st.pyplot()

    plt.scatter(selected_data_std_pca['pca1'], selected_data_std_pca['pca3'], c=meanshift.labels_, cmap='Paired', alpha=0.8)
    st.pyplot()

    plt.scatter(selected_data_std_pca['pca2'], selected_data_std_pca['pca3'], c=meanshift.labels_, cmap='Paired', alpha=0.8)
    st.pyplot()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax =plt.scatter(selected_data_std_pca['pca1'], selected_data_std_pca['pca2'], selected_data_std_pca['pca3'], c=meanshift.labels_,cmap='Paired' )
    st.pyplot()

#ANN
def aNN(real_data):
    X = real_data.iloc[:, 1:4].values # we want only the columns from 3 to 12, columns 0,1 and 2 are not necessary
    y = real_data.iloc[:, 4].values
    enc = OneHotEncoder()
    Y = enc.fit_transform(y[:, np.newaxis]).toarray()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data set into training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y, test_size=0.3)


    n_features = X.shape[1]
    n_classes = Y.shape[1]
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)

    from keras.models import Sequential
    from keras.layers import Dense




    def create_custom_model(input_dim, output_dim, nodes, n=1, name='model'):
        def create_model():
            # Create model
            model = Sequential(name=name)
            for i in range(n):
                model.add(Dense(nodes, input_dim=input_dim, activation='relu'))
            model.add(Dense(output_dim, activation='softmax'))

            # Compile model
            model.compile(loss='categorical_crossentropy', 
                        optimizer='adam', 
                        metrics=['accuracy'])
            return model
        return create_model

    models = [create_custom_model(n_features, n_classes, 10, 1, 'model_{}'.format(1))]
    for create_model in models:
        create_model().summary()
    from keras.callbacks import TensorBoard

    history_dict = {}

    # TensorBoard Callback
    cb = TensorBoard()

    for create_model in models:
        model = create_model()
        print('Model name:', model.name)
        history_callback = model.fit(X_train, Y_train, batch_size=10, epochs=100, validation_data=(X_test, Y_test), callbacks=[cb])
        score = model.evaluate(X_test, Y_test)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        
        history_dict[model.name] = [history_callback, model]
    training_result = model.predict(X_train)
    pred_result = model.predict(X_test)
    prd = np.concatenate((training_result, pred_result))
    real_all = np.concatenate((Y_train, Y_test))
    pred = list()
    for i in range(len(prd)):
        pred.append(np.argmax(prd[i]))
        
    real = list()
    for i in range(len(real_all)):
        real.append(np.argmax(real_all[i]))
    
    result_clustering=pd.DataFrame()
    result_clustering['real_label']=real
    result_clustering['pred_label']=pred
    result_clustering['result'] = np.where(result_clustering["real_label"] == result_clustering["pred_label"], True, False)
    result_clustering.loc[result_clustering['result']==False]

    import matplotlib.pyplot as plt
    plt.plot(history_callback.history['accuracy'])
    plt.plot(history_callback.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    st.pyplot()

    plt.plot(history_callback.history['loss']) 
    plt.plot(history_callback.history['val_loss']) 
    plt.title('Model loss') 
    plt.ylabel('Loss') 
    plt.xlabel('Epoch') 
    plt.legend(['Train', 'Test'], loc='upper left') 
    st.pyplot()

#DNN
def dNN(train_data):
    training_data = train_data.iloc[:, 1:4].values
    training_labels= train_data.iloc[:, 4].values
    enc = OneHotEncoder()
    Y = enc.fit_transform(training_labels[:, np.newaxis]).toarray()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(training_data)

    X, testX, Y, testY = train_test_split(
        X_scaled, Y, test_size=0.3)


    # Parameters
    learning_rate = 0.005
    training_epochs =100
    batch_size = 100
    display_step = 1


    # Network Parameters
    n_hidden_1 = 100 # 1st layer number of features
    n_hidden_2 = 100 # 2nd layer number of features
    n_input =X.shape[1]
    #n_classes = 4 # MNIST total classes (0-9 digits)

    #n_features = X.shape[1]
    n_classes = Y.shape[1]


    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])
    # Create model
    def multilayer_perceptron(x, weights, biases):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer


    # In[24]:

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    # Construct model
    pred = multilayer_perceptron(x, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.initialize_all_variables()

    t1_2 = datetime.datetime.now()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)


        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(X.shape[0]/batch_size)
            i=0
            while i < total_batch:
                start = i
                end = i+ batch_size
                    
                        
                batch_x = np.array(X[start:end])
                batch_y = np.array(Y[start:end])
                        

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                            y: batch_y})
                avg_cost += c/total_batch
            
                i+=batch_size
            
            # Display logs per epoch step
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost=",             "{:.9f}".format(avg_cost))
                
        print ("Optimization Finished!")



            #  model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            # Size of dataset
        st.write( "Data set:", X.shape[0]+testX.shape[0])
        st.write ("Training set:", X.shape[0])
        st.write ("Test set:", testX.shape[0])
        st.write ("Featuires:",X.shape[1])
        
        
            # Calculate accuracy
        
        st.write ("Training Accuracy:", accuracy.eval({x:X, y:Y}))
        
        st.write ("Test Accuracy:", accuracy.eval({x:testX, y:testY}))
    
        pred_test=tf.argmax(pred,1)
        keep_pred=pred_test.eval(feed_dict={x:testX},session=sess)
        training_pred=pred_test.eval(feed_dict={x:X}, session=sess)
        
        
        training_result=np.asarray(training_pred)
        pred_result=np.asarray(keep_pred)
        
        
        real_result=np.asarray(Y)
        real_test_result=np.asarray(testY)
            
            
        np.savetxt("pred_result_dnn.csv", pred_result,delimiter=",",header="Prediction",fmt="%.0f")
        np.savetxt("training_pred_result_dnn.csv", training_result, delimiter=",", header="Prediction", fmt="%.0f")   
        np.savetxt("training_real_result_dnn.csv", real_result, delimiter=",", header="Class", fmt="%.0f") 
        np.savetxt("test_real_result_dnn.csv",real_test_result, delimiter=",", header="Class", fmt="%.0f")
    
    t2_2 = datetime.datetime.now()

    print ("Computation time: " + str(t2_2 - t1_2))
    prd = np.concatenate((training_result, pred_result))
    real_all = np.concatenate((Y, testY))
    real = list()
    for i in range(len(real_all)):
        real.append(np.argmax(real_all[i]))
    result_clustering = pd.DataFrame()
    result_clustering['DNN_pred'] = prd
    result_clustering['DNN_real'] = real
#mpl
def mPL(x_train, y_train, x_test, y_test, st):
    from sklearn.neural_network import MLPClassifier

    model = MLPClassifier()
    model.fit(x_train, y_train)

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    mlp_train_score = model.score(x_train, y_train)
    mlp_test_score = model.score(x_test, y_test)

    st.write("Training Accuracy :", model.score(x_train, y_train))
    st.write("Testing Accuracy :", model.score(x_test, y_test))

    # cm = confusion_matrix(y_test, pred_test)
    # print(cm)