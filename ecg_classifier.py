# =============================================================================
# 
# CORMEUM ecg classifier
# 
# =============================================================================

import preprocessing
import feature_extraction
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier


__DATA_DIR = 'd://training2017'
FREQUENCY = 300


# =============================================================================
# 1. Preprocessing
# =============================================================================


x, y = preprocessing.load_data(__DATA_DIR)
y = preprocessing.format_labels(y)
subX, subY = preprocessing.balance(x, y)
subX = [preprocessing.normalize_ecg(i) for i in subX]


# =============================================================================
# 2. Feature extraction
# =============================================================================


print('number of columns before feature extraction : {}'.format(len(x[0])))

fn = feature_extraction.get_feature_names(subX[0])
subX = [feature_extraction.features_for_row(i) for i in subX]

print('number of columns after feature extraction : {}'.format(subX[0].shape))


# =============================================================================
# 3. Build a ML model
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(subX, subY, test_size=0.2)

#param_grid = {'max_depth':  [1, 2, 5, 10, 20, 50, 100], 'max_features': [1, 2, 5, 10, 30, 50]}
param_grid = {'max_depth': [50], 'max_features': [30]}
grid = GridSearchCV(RandomForestClassifier(n_estimators=1000, 
                                           n_jobs=-1
                                           ), param_grid=param_grid, cv=5)
                                            
grid.fit(X_train, y_train)