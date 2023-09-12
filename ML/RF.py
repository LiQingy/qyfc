def ML_rf(feature, target):
    '''
    Random forest

    Parameters:
    ===========
    feature: 
    target:

    '''
    import numpy as np 
    import h5py

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split


    x_train,x_test,y_train,y_test = train_test_split(feature, target, test_size=0.3, random_state=0)

    forest = RandomForestRegressor(n_estimators=400, min_samples_leaf = 5, random_state=0, oob_score = True)
    forest.fit(x_train, y_train)

    #OOB score (R^2 for out-of-bag data)
    score = forest.oob_score_

    #prediction
    y_pred = forest.predict(x_test)




    # check for trees and OOB sample
    # from sklearn import tree
    # for trees in forest.estimators_:
    #     tree.plot_tree(trees)
    #     # Here at each iteration we obtain out of bag samples for every tree.
    #     unsampled_indices =_generate_unsampled_indices(tree.random_state,n_samples,n_samples)
    #     unsampleX=x_train[unsampled_indices,:]
    #     unsampley=y_train[unsampled_indices]
    #     print(tree.decision_path(x_train))
        
    #     predictions[unsampled_indices] += tree.predict(unsampleX)
    #     n_predictions[unsampled_indices] += 1
    #     sktree.plot_tree(tree)





