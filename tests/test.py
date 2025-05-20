#create tests to check whether the read in was successful - that the df is not empty
#create test that check whether the amount of features we specify in config is the same as number of columns in the df
#test to check for data types
#check on amount of nas

def test_na_amount():
    assert df[config.target].isnull().sum() == 0

def test_col_amount():
    selected_cols = config.cat_features + config.num_features + [config.target] + ['Booking_ID']
    assert len(df.columns)==len(selected_cols)

    #test that a df was written to UC