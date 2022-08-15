# PHYAuthentication
multi node PHY authentication with DL

 1) run data_generator.py to create a dataset 
    dist=[x1, x2, x3, ..., xn, xs] vector indicates that we are generating data 
    for n legitimate nodes located at distances x1,x2, ..., xn meters and a spoofer located at xs meters

 2) run main2.py to train the DL model, test it for both DL and traditional models
    traditional model is run for a range of different thresholds
    Precision, Recall and Accuracy is checked and saved to a txt file
