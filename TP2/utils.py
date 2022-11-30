def tipo_propiedad_train(property_type):
    '''Evaluación del modelo sobre conjunto de datos de train'''
    
    property_type = prop_prueba.loc[prop_prueba['property_type'] == property_type]
    
    feature_cols = [x for x in property_type.columns if ((x != 'property_type') & (x != 'price_aprox_usd') & (x != 'place_name'))]
    
    # Generando Train y Test 
    X = property_type[feature_cols]
    y = property_type['price_aprox_usd']

    # División de los datos en train y test
    # ==============================================================================    
    X_train, X_test, y_train, y_test = train_test_split(
                                        X,
                                        y,
                                        train_size   = 0.8,
                                        random_state = 123,
                                     ) 

    # Generando modelo 

    X = X_train[feature_cols]
    y = y_train

    # Tenemos que agregar explícitamente una constante:
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    return model.summary()
