import logging
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyranges as pr
import seaborn as sns
from scipy.stats import spearmanr, zscore
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    roc_auc_score,
    f1_score,
)
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
import xgboost as xgb
from modality.annotation import get_transcripts




# setup
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

sns.set_theme()
sns.set_style("whitegrid")
biomodal_palette = ["#003B49", "#9CDBD9", "#F87C56", "#C0DF16", "#05868E"]
sns.set_palette(biomodal_palette)
column_order = [
    'before_tss', 'after_tes', 'five_prime_utrs', 'first_exons', 'first_introns', 
    'exons', 'introns', 'three_prime_utrs', 'genes',
    ]


######## DATA LOADING ########
def select_transcript_based_on_tag(df):
    # for each transcript in df, select the one with the highest priority tag
    # priorities are:
        # 1. 'basic,appris_principal_1,CCDS'
        # 2. 'basic,appris_principal_1'
        # 3. 'basic,CCDS'
        # 4. 'basic'
    # but with 'exp_conf' (experimentally confirmed) tag, the priority is higher.
    
    priorties = {
        'basic,appris_principal_1,exp_conf,CCDS': 1,
        'basic,appris_principal_1,CCDS': 1,
        'basic,appris_principal_1,exp_conf': 3,
        'basic,appris_principal_1': 4,
        'basic,exp_conf,CCDS': 5,
        'basic,CCDS': 6,
        'basic,exp_conf': 7,
        'basic': 8
    }

    # sort the dataframe by the priority of the tags
    df['tag_priority'] = df.tag.map(priorties)

    df = df.sort_values(by='tag_priority')

    # drop duplicates, keeping the first one
    df = df.drop_duplicates(subset='gene_id', keep='first')

    return df[["gene_id", "transcript_id"]]

def tpm_to_rpkm(tpm, total_reads, transcript_length):
    """
    Convert TPM to RPKM.

    Parameters:
    tpm (float or array): TPM values
    total_reads (float): Total number of reads in the RNA-Seq experiment
    transcript_length (float or array): Transcript length in base pairs

    Returns:
    rpkm (float or array): RPKM values
    """
    transcript_length_kb = transcript_length / 1000  # Convert to kilobases
    rpkm = (tpm * total_reads) / transcript_length_kb
    return rpkm

def load_rna_expression(features, selected_transcripts, transcripts):
    ge_path = "../data/quant.sf"
    df_expression = pd.read_csv(ge_path, sep="\t")
    print(df_expression.shape)

    full_selected_transcript = pd.merge(selected_transcripts,transcripts, left_on='transcript_id', right_on='id', how='left')

    # Merge df_expression with transcripts on the matching columns
    merged_df = pd.merge(full_selected_transcript[['id', 'parent', 'strand']], df_expression, left_on='id', right_on='Name', how='left')
    merged_df = pd.merge(merged_df, features[['Gene_id','contig']], left_on='parent', right_on='Gene_id', how='left')

    # Add the 'gene_id' column to df_expression
    df_expression['Gene_id'] = merged_df['parent']
    df_expression['strand'] = merged_df['strand']
    df_expression['contig'] = merged_df['contig']
    df_expression_clean = df_expression[~df_expression.isna().any(axis=1)]
    df_expression_clean

    return df_expression_clean


######## END OF DATA LAODING ########



######## EXPLORATORY DATA ANALYSIS ########
def plot_target(df, target):
    plt.figure(figsize=(10, 6))
    plt.scatter(df.index, df[target], color='blue', alpha=0.6)
    plt.xlabel('Index')
    plt.ylabel('Label')
    plt.title('Scatter Plot of Label Across Index')
    plt.show()

def plot_target_kde(df,target):
    sns.kdeplot(np.log2(df[target]+0.001))

def plot_numeric_features_histogram_grid(df, target, num_cols=4):
    # List of numeric features to plot
    # features = [col for col in df.columns if df[col].dtype != 'object']
    features = [col for col in df.columns if col != target and col != "strand"]
    num_features = len(features)
    print(f"number of features: {num_features}")

    
    # Calculate number of rows needed for the grid
    num_rows = int(np.ceil(num_features / num_cols))
    
    # Create subplots with grid layout
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols * 5, num_rows * 4))
    
    # Flatten axes array if needed
    axes = axes.flatten() if num_features > 1 else [axes]
    
    # Plot histograms for each feature
    for i, feature in enumerate(features):
        sns.histplot(data=df, x=feature, ax=axes[i], kde=True)
        axes[i].set_title(f'Distribution of {feature}')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Count')
    
    # Hide any unused subplots
    for ax in axes[num_features:]:
        ax.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

def print_high_corr(features_df):
        # Compute the correlation matrix
    corr_matrix = features_df.corr()

    # Get the absolute values of the correlation matrix
    corr_matrix_abs = corr_matrix.abs()

    # Mask the upper triangle of the matrix (to remove mirrored pairs)
    mask = np.triu(np.ones_like(corr_matrix_abs, dtype=bool))

    # Apply the mask to the correlation matrix
    reduced_corr_matrix = corr_matrix_abs.mask(mask)

    # Unstack the matrix and drop NaN values (resulting from the mask)
    corr_pairs = reduced_corr_matrix.unstack().dropna()

    # Filter the pairs with a correlation higher than 0.5
    high_corr_pairs = corr_pairs[corr_pairs > 0.5]

    # Sort the pairs by correlation value
    high_corr_pairs_sorted = high_corr_pairs.sort_values(ascending=False)

    # Display the unique pairs with correlation higher than 0.5
    print(high_corr_pairs_sorted)
    print(f"{len(high_corr_pairs)} / {len(corr_pairs)}")





######## END OF EXPLORATORY DATA ANALYSIS ########



######## DATA PREPROCESSING ########

def remove_target_outliers(df, target, threshold):
    # Calculate Z-scores
    z_scores = np.abs(zscore(df[target]))
    # Filter out outliers
    print(f"number of entries removed: {df[(z_scores >= threshold)].shape[0]} (out of {df.shape[0]})")
    df_clean = df[(z_scores < threshold)]
    plot_target(df_clean, target)
    return df_clean

def remove_obvious_outliers(df):
    """ remove feature outliers """
    mc_cols = [col for col in df.columns if col.startswith('mean_mc')]
    hmc_cols = [col for col in df.columns if col.startswith('mean_hmc')]
    suspicious_mc_rows = df[(df[mc_cols] == 1.0).any(axis=1)]
    suspicious_hmc_rows = df[(df[hmc_cols] == 1.0).any(axis=1)]
    # Combine the two groups to find all suspicious rows
    suspicious_rows = pd.concat([suspicious_mc_rows, suspicious_hmc_rows]).index.unique()
    print(f"rows with 1.0 mc / hmC mean meth fration: {len(suspicious_rows)}")

    # Drop the suspicious rows from the original DataFrame
    df_clean = df.drop(suspicious_rows)
    return df_clean

def impute_multimodal(data_train, data_test, columns_to_keep, missing_values_strategy="mean"):
    print(f"inside impute -- train: {data_train.shape}, test: {data_test.shape}")

    # Subset the columns to be used
    train_subset = data_train[columns_to_keep].copy()
    test_subset = data_test[columns_to_keep].copy()

    print(f"subsets: {data_train.shape}, {data_test.shape}")

    # Apply KMeans clustering on the training data (on non-missing values)
    kmeans = KMeans(n_clusters=2, random_state=42)
    non_missing_train = train_subset.dropna()
    kmeans.fit(non_missing_train)
    # Predict clusters for the training data
    train_subset['cluster'] = kmeans.predict(train_subset.fillna(0))

    # Assign clusters to the test data based on the fitted model
    test_subset['cluster'] = kmeans.predict(test_subset.fillna(0))  # Avoid issues by filling NaNs temporarily

    print("initialised clusters")
     # Impute within each cluster for the training set
    for cluster in range(kmeans.n_clusters):
        # Train set: select rows that belong to the current cluster
        cluster_train_data = train_subset[train_subset['cluster'] == cluster]
        
        # Impute missing values for the current cluster in the training set
        imputer = SimpleImputer(strategy=missing_values_strategy)
        imputed_train = imputer.fit_transform(cluster_train_data.drop(columns='cluster'))
        train_subset.loc[train_subset['cluster'] == cluster, columns_to_keep] = imputed_train

    # Impute within each cluster for the test set
    for cluster in range(kmeans.n_clusters):
        # Test set: select rows that belong to the current cluster
        cluster_test_data = test_subset[test_subset['cluster'] == cluster]
        
        # Use the same imputer fitted on the train cluster to impute test set
        imputed_test = imputer.transform(cluster_test_data.drop(columns='cluster'))
        test_subset.loc[test_subset['cluster'] == cluster, columns_to_keep] = imputed_test

    # Drop the 'cluster' column after imputation
    train_subset = train_subset.drop(columns='cluster')
    test_subset = test_subset.drop(columns='cluster')

    # Update the original data
    data_train[columns_to_keep] = train_subset
    data_test[columns_to_keep] = test_subset

    return data_train, data_test

def select_features(features, mod):
    """
    Select features based on the modification type. Only keep the features corresponding to the list `mod`. 
    E.g. if mod="modc", only keep features related to modC, and discard those related to mC and hmC
    """
    if isinstance(mod, str):
        mod = [mod]
    return [f for f in features if any([m in f for m in mod]) or (("cpg_count" in f) or ("range" in f) or ("strand" in f))]

def stratified_sampling_per_contig(df, target, test_size=0.1):
    test_indices = []

    # Group by contig
    grouped = df.groupby('Chromosome')
    
    for contig, group in grouped:
        # Split the group into train and test
        # group = group.sample(frac=1, random_state=42).reset_index(drop=False)
        
        train, test = train_test_split(
            group,
            #stratify=group[target],
            test_size=test_size,
            random_state=42
        )
        
        test_indices.extend(test.index)
    
    test_set = df.loc[test_indices]
    train_set = df.drop(test_indices)
    print(f"trainset: {train_set.shape}")
    print(f"testset: {test_set.shape}")
    return train_set, test_set

def split_train_test_data(data, features, target, test_contig=None, missing_values_strategy="impute_mean"):
    """
    Split the data into training and testing sets using the specified test contig
    """
    print("inside train test split")
    if isinstance(test_contig, str):
        test_contig = [test_contig]
    # data_train = data[~data["Chromosome"].isin(test_contig)]
    # data_test = data[data["Chromosome"].isin(test_contig)]
    data_train , data_test = stratified_sampling_per_contig(data, target) #TODO:need to bin target to stratify

    # Check the distribution of TPM in the test set
    print(data_test[target].value_counts()/len(data))
    print('-'*50)
    # Check the distribution of contigs in the test set
    print(data_test['Chromosome'].value_counts()/data['Chromosome'].value_counts())
    print('-'*50)

    print("before imputing")

    # Assuming 'features' is your list of feature names
    impute_features = [f for f in features if not f.startswith("strand")]

    data_train, data_test = impute_multimodal(
        data_train,
        data_test,
        impute_features + [target],
        "mean"
    )
    # data_train, data_test = impute_missing_values(
    #     data_train, 
    #     data_test, 
    #     impute_features + [target], 
    #     missing_values_strategy
    #     )
    X_train, y_train = data_train[features], data_train[target]
    X_test, y_test = data_test[features], data_test[target]
    return X_train, X_test, y_train, y_test


def scale_counts(df,method="StandardScaler"):
    columns_to_scale = [col for col in df.columns if col.startswith("range_length") or col.startswith("cpg_count")]
    if method == "StandardScaler":
      scaler = StandardScaler()
    elif method == "MinMaxScaler":
        scaler = MinMaxScaler()
    else:
        print("not scaled!!!")
        exit=1
    # Apply the scaler on the selected columns
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    return df

def log_transform(data):
    transformed = data.copy()
    if type(data) == pd.DataFrame:
        columns_to_transform = [col for col in data.columns if col.startswith("mean_")]
        # Apply log2(x + 0.001) to the specified columns
        transformed[columns_to_transform] = data[columns_to_transform].apply(lambda x: np.log2(x + 0.001))

    elif type(data) == pd.Series:
        transformed= np.log2(data+0.001)

    return transformed


def preprocess(data, 
  features, 
  mod,
  target, 
  missing_values_strategy="impute_mean", 
  test_contig=None, 
):

  features = select_features(features, mod)
  print(features)

  data_cleaned = remove_obvious_outliers(data)
  print(data_cleaned.head())

  X_train, X_test, y_train, y_test = split_train_test_data(
        data_cleaned, 
        features, 
        target, 
        test_contig,
        missing_values_strategy,
        )


  print("Shape of train data")
  X_train = scale_counts(X_train,"StandardScaler")
  # X_train = log_transform(X_train)
  y_train = log_transform(y_train)
  print(X_train.shape)

  print("Shape of test data")
  X_test = scale_counts(X_test, "StandardScaler")
  # X_test = log_transform(X_test)
  y_test = log_transform(y_test)
  print(X_test.shape)


  return X_train, X_test, y_train, y_test


######## END OF DATA PREPROCESSING ########




######## MODELLING ########

# regression
def tune_parameters(X_train, y_train, model=xgb.XGBRegressor(eval_metric="rmsle"), scoring="r2"):
    """
    Identify the best hyperparameters of the XGBoost regressor using GridSearchCV
    """
    param_grid = {
        "max_depth": [5, 6, 7],
        "n_estimators": [600, 800, 1000], #[200, 300, 400, 500, 600, 700, 800],
        "subsample": [0.6, 0.7, 0.8],
        "min_child_weight": [1, 5, 10],
        "colsample_bytree": [0.8, 0.9],#[0.75, 0.8, 0.85, 0.9],
        "eta": [0.01, 0.02, 0.03]#[0.01, 0.02, 0.03, 0.04, 0.05],
    }
    
    search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=4).fit(X_train, y_train)
    return search.best_params_

# regression
def run_regressor(
        X_train, X_test, y_train, y_test,
        hyperparameters,
        random_state=1, 
        find_optimal_parameters=False
        ):
    """
    Run the XGBoost regressor using the specified data and parameters
    """

    model = xgb.XGBRegressor(eval_metric="rmsle")
    scoring="r2"
  
    if find_optimal_parameters:
                best_params = tune_parameters(X_train, y_train, model, scoring)
                hyperparameters.update(best_params)

    print("Using params")
    print(hyperparameters)
    

    model = train_regression_model(X_train, y_train, hyperparameters, random_state)

    mse, rmse, mae, r2, spear_r = evaluate_model(model, X_test, y_test)

    df_metrics = pd.DataFrame({
        "mse": [mse],
        "rmse": [rmse],
        "mae": [mae],
        "r2": [r2],
        "spearman": [spear_r[0]],
    })

    y_pred = model.predict(X_test)

    return model, df_metrics, y_test, y_pred

# regression
def train_regression_model(X_train, y_train, hyperparameters, random_state=1):
    """ 
    Train an XGBoost regressor using the specified hyperparameters
    """
    regressor = xgb.XGBRegressor(
        random_state=random_state,
        **hyperparameters,
    )
    regressor.fit(X_train, y_train)
    return regressor

# classifier
def run_classifier(
    X_train, X_test, y_train, y_test,
    hyperparameters,
    find_optimal_parameters,
    random_state=1,
    ):
    """
    Run an xgboost classifier
    """
    scoring = "roc_auc"

    classifier = xgb.XGBClassifier(eval_metric=scoring)

    if find_optimal_parameters:
        
        best_params = tune_parameters(X_train, y_train, classifier, scoring)
        hyperparameters.update(best_params)

    print("Using params")
    print(hyperparameters)

    classifier = train_classifier_model(X_train, y_train, hyperparameters, random_state)

    accuracy, f1, auc = evaluate_classifier(classifier, X_test, y_test)

    df_metrics = pd.DataFrame(
        {
            "number_of_categories": [len(np.unique(y_train))],
            "accuracy": [accuracy],
            "macro_f1": [f1],
            "auc": [auc],

        }
    )
    return classifier, df_metrics


# classifier
def train_classifier_model(X_train, y_train, hyperparameters, random_state=1):
    """
    Train an XGBoost classifier using the specified hyperparameters
    """
    classifier = xgb.XGBClassifier(
        random_state=random_state,
        **hyperparameters,
    )
  
    print("Using params")
    print(hyperparameters)

    classifier.fit(X_train, y_train)
    return classifier



######## END OF MODELLING ########


######## EVALUTATION ########

# regression
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the performance of the model using the test set
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    spear_r = spearmanr(y_test, y_pred)
    return mse, rmse, mae, r2, spear_r

# classifier
def evaluate_classifier(model, X_test, y_test):
    """
    Evaluate the performance of the classifier using the test set
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred, average="macro")

    # calculate AUC
    if len(np.unique(y_test)) == 2:
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    else:
        auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr", average="macro")

    return accuracy, f1, auc

def plot_results(y_test, y_pred, title, target):
    """
    Plot the observed vs predicted expression values for the test set
    """
    biomodal_palette = ["#003B49", "#9CDBD9", "#F87C56", "#C0DF16", "#05868E"]
    plt.plot(y_test, y_pred, ".", ms=4, c=biomodal_palette[0])
    # add x=y line
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "--", color=biomodal_palette[2])
    plt.xlabel(f"Observed Expr. {target}")
    plt.ylabel(f"Predicted Expr. {target}")
    plt.title(title)
    plt.xlim(min(y_test)-2, max(y_test)+2)
    plt.ylim(min(y_pred)-2, max(y_pred)+2)
    plt.grid(True)
    plt.show()

######## END OF EVALUTATION ########


######## FEATURE IMP ########

def evaluate_feat_imp(best_model, features):
    df_features_importance = pd.DataFrame(
        {
            "feature": select_features(features, ["mc", "hmc"]),
            "importance": best_model.feature_importances_,
        }
    
    )
    print(df_features_importance.sort_values("importance", ascending=False).head(10))

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(
        x="feature", 
        y="importance", 
        data=df_features_importance.sort_values("importance", ascending=False),
        )
    plt.xticks(rotation=60, ha="right")
    plt.show()
    
######## END OF FEATURE IMP ########


######## UTILITY FUNCTIONS ########

def build_regression_model(X_train, X_test, y_train, y_test, target, default_hyperparameters, find_optimal_parameters=False):
    
    model, df_metrics, y_test, y_pred = run_regressor(
        X_train, X_test, y_train, y_test,
        hyperparameters=default_hyperparameters,
        random_state=0,
        find_optimal_parameters=find_optimal_parameters,
    )  

    print(df_metrics)
    plot_results(
        y_test, 
        y_pred, 
        f"Bulk RNA-seq - R^2={df_metrics.r2.values[0]:.2f}, Spearman R={df_metrics.spearman.values[0]:.2f},",
        target
    )

def build_classification_model(X_train, X_test, y_train, y_test, default_hyperparameters, n_cat, find_optimal_parameters=False):
    # create n categories based on the response values
    labels = [k for k in range(n_cat)]
    # df_features_expression["category"] = pd.cut(
    #     df_features_expression["Response"], bins=n, labels=labels
    # )
    y_train_binned = pd.cut(
        y_train, bins=n_cat, labels=labels
    )
    y_test_binned = pd.cut(
        y_test, bins=n_cat, labels=labels
    )
    
    # run the classifier
    c, df_metrics = run_classifier(
        X_train, X_test, y_train_binned, y_test_binned,
        hyperparameters=default_hyperparameters,
        find_optimal_parameters=find_optimal_parameters,
        random_state=1,
        )
    return c, df_metrics



######## END oF UTILITY FUNCTIONS ########

def run_main(input, target):
    # feature output from extract_feature.ipynb
    df_features = pd.read_pickle(input)
    print(f"{input} features loaded")

    # Get transcripts for mm10
    print("loading gene annotation mm10")
    transcripts = get_transcripts(
            reference="mm10",
            contig=None,
            start=None,
            end=None,
            as_pyranges=False,
        )

    selected_transcripts = transcripts.groupby('gene_id').apply(
        select_transcript_based_on_tag
        ).reset_index(drop=True)
    print(f"transcript downloaded, selected transcripts: {selected_transcripts}")
   
    # expression data
    df_expression = load_rna_expression(df_features, selected_transcripts, transcripts)
    print("TPM expression data loaded")
    
    df_features_expression = pd.merge(
        df_features,
        df_expression,
        on=["Gene_id", "contig", "strand"],
        how="inner",
    )
    print(f"features and target merged: {df_features_expression}")

    print('='*200)
    
    
    # preprocessing:
    orig_feature_shape = df_features_expression.shape
    # drop columns with Nan
    df_features_expression_clean = df_features_expression.dropna(axis=1, how='all')
    # frop contig column
    df_features_expression_clean = df_features_expression_clean.drop('contig', axis=1)
    print(f"Nan features dropped: {orig_feature_shape[1]} --> {df_features_expression_clean.shape[1]}")

    # add column rkpm for alternative prediction target
    if target == "rkpm":
        df_features_expression_clean["rkpm"] = df_features_expression_clean.apply(
            lambda x: tpm_to_rpkm(x['TPM'], x['NumReads'], x['EffectiveLength']),
            axis=1
        )

    print(f"selected target unit: {target}")
    print("PREPROCESSING...")

    zscore_threshold = 0.5
    print(f"removing rows with {target} zscore > {zscore_threshold} - target outliers")
    df_clean = remove_target_outliers(df_features_expression_clean, target, zscore_threshold)
    # Label Encode strand: '+' -> 0 ; '-' -> 1
    df_clean['strand'] = LabelEncoder().fit_transform(df_clean['strand'])
    
    nan_columns = df_clean.isna().all() #double checks nan columns are dropped
    print(f"Columns with nan values:{nan_columns[nan_columns].index}")

    df_clean = df_clean.drop(['range_length_before_tss_sense', 'range_length_after_tes_sense'], axis=1)

    features = [
    c for c in df_clean.columns if
        (c.startswith("mean")
        or c.startswith("cpg_count")
        or c.startswith("range")
        or c.startswith("strand"))
    ]
    print(f"Training with {len(features)} features...")

    # train-test split
    X_train, X_test, y_train, y_test = preprocess(
    data=df_clean,
    features=features,
    mod = ["mc", "hmc"],
    target=target,
    missing_values_strategy="impute_mean",
    )

    default_hyperparameters = {
        'n_estimators': 600, 
        'colsample_bytree': 0.8, 
        'eta': 0.01, 
        'max_depth': 5, 
        'min_child_weight': 10, 
        'subsample': 0.8
    }

    # Building a regressor:
    regression_gridsearch = False
    if regression_gridsearch:
        print("grid serching (Regression)")
    else:
        print(f"training with {default_hyperparameters}")

    build_regression_model(X_train, X_test, y_train, y_test, target, default_hyperparameters, regression_gridsearch) #skip gridsearch for now
    
    
    print('='*200)
    # Build a classifier:
    classifier_gridsearch = False
    
    number_of_categories = np.arange(2, 6, 1)
    classifiers = {}
    classifier_res = pd.DataFrame(columns=["number_of_categories", "accuracy", "macro_f1", "auc"])


    # Loop through categories
    for n in number_of_categories:
        if regression_gridsearch:
            print(f"training {n} category model with grid serching (classification)")
        else:
            print(f"training {n} category model with {default_hyperparameters}")
        
        classifier, df_metrics = build_classification_model(X_train, X_test, y_train, y_test, default_hyperparameters, n, classifier_gridsearch)
        
        # Store the classifiers and metric for later use
        classifiers[n] = classifier
        classifier_res = pd.concat([classifier_res, df_metrics], ignore_index=True)

    # Print all results
    print("classifier results:")
    print(classifier_res)
    print('='*200)

    # Find the best-performing model (e.g., based on accuracy)
    best_model_id = classifier_res.loc[classifier_res["auc"].idxmax()]["number_of_categories"]
    best_model = classifiers[best_model_id]

    print(f"Best performing classifier: {best_model_id} categories")


    # Feature Importance:
    evaluate_feat_imp(best_model, features)




