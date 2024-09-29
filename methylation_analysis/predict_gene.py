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
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE
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

def aggregate_expression(df_expression, transcripts, df_features):
    # Merging and calculating the summed TPM values in one line
    df_features["TPM"] = df_features["Gene_id"].map(
        df_expression
        .merge(transcripts[['id', 'parent', 'strand']], left_on='Name', right_on='id', how='left')
        .groupby('parent')['TPM']
        .sum()
    )

    # Filtering out rows with NaN values in the 'TPM' column
    return df_features[~df_features['TPM'].isna()]
                         

def load_rna_expression_tpm(rna_input, features, transcripts, expressed_only=False):
    df_expression = pd.read_csv(rna_input, sep="\t")
    # print(f"all transcripts {transcripts.shape}")
    # full_selected_transcript = pd.merge(selected_transcripts,transcripts, left_on='transcript_id', right_on='id', how='right')
    # print(f"full_selected_transcript {full_selected_transcript.shape}")
    # # Merge df_expression with transcripts on the matching columns
    # merged_df = pd.merge(full_selected_transcript[['id', 'parent', 'strand']], df_expression, left_on='id', right_on='Name', how='left')
    # merged_df = pd.merge(merged_df, features[['Gene_id','contig']], left_on='parent', right_on='Gene_id', how='left')
    
    #ignore selected transcripts, aggregate all expression quantity across isoforms
    df_features_expression = aggregate_expression(df_expression, transcripts, features)

    
    # Add the 'gene_id' column to df_expression
    # df_expression['Gene_id'] = merged_df['parent']
    # df_expression['strand'] = merged_df['strand']
    # df_expression['contig'] = merged_df['contig']
    # df_features_expression = df_features_expression[~df_features_expression.isna().any(axis=1)]
    # df_features_expression = df_features_expression.dropna(axis=1, how='all')

    print('target loaded, shape:')
    print(df_features_expression.shape)

    if expressed_only:
        df_features_expression = df_features_expression[df_features_expression["TPM"]>0]
        print('target filtered, TPM>0, shape:')
        print(df_features_expression.shape)

    return df_features_expression

def rpm_to_rpkm(rpm, length):
    """
    Convert RPM to RPKM
    """
    return rpm - np.log2(length) + np.log2(1e3)

def load_rna_expression_rpm(rna_data_path, selected_transcripts, transcripts, target, convert_to_rpkm=True):
    
    df_expression = pd.read_csv(rna_data_path, sep="\t")

    # full_selected_transcript = pd.merge(selected_transcripts, transcripts[['id','gene_name']], left_on='transcript_id', right_on='id', how='left')

    # revert from log2(RPM) to RPM before taking the mean
    df_expression["E14_rep_1"] = np.power(2, df_expression["E14_rep_1 (log2 RPM)"])
    df_expression["E14_rep_2"] = np.power(2, df_expression["E14_rep_2 (log2 RPM)"])
    df_expression["E14_rep_3"] = np.power(2, df_expression["E14_rep_3 (log2 RPM)"])

    # take the mean of the three replicates
    df_expression["E14_expr"] = np.mean(
        df_expression[["E14_rep_1", "E14_rep_2", "E14_rep_3"]],
        axis=1,
    )

    # log2 transform
    # df_expression[target] = np.log2(df_expression["E14_expr"]+0.001)
    df_expression[target] = df_expression["E14_expr"]

    # convert to RPKM
    if convert_to_rpkm:
        df_expression["Gene_length"] = df_expression["End"] - df_expression["Start"]
        df_expression[target] = rpm_to_rpkm(
            df_expression[target], df_expression["Gene_length"]
        )

    # Convert to 0-based
    df_expression["Start"] = df_expression["Start"] - 1
    df_expression["End"] = df_expression["End"] - 1

    print("rpkm dataset shape:")
    print(df_expression.shape)
    # df_expression = pd.merge(full_selected_transcript, df_expression, left_on='gene_name', right_on='Gene', how='inner')
    

    return df_expression


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
    min_expressed=df[df[target]!=0][target].min()
    print(f"minimum non-zero value: {min_expressed}, using {min_expressed/100} as a constant for log transform [plotting only]")
    sns.kdeplot(np.log2(df[target]+min_expressed/100))
    plt.show()
    # sns.kdeplot(df[target])
    

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


def impute_multimodal(data_train, data_test, columns_to_impute, missing_values_strategy="mean"):
    print(f"inside impute -- train: {data_train.shape}, test: {data_test.shape}")

    # Iterate through each column that needs imputation
    for column in columns_to_impute:

        # Extract non-missing values for the column in the training set
        non_missing_train = data_train[column].dropna().values.reshape(-1, 1)
        
        if non_missing_train.size == 0:
            print(f"No non-missing values in column {column}.")
            continue  # Skip if no non-missing values

        # Apply KMeans clustering on the non-missing values
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(non_missing_train)

        # Predict clusters for both train and test data
        data_train['cluster'] = kmeans.predict(data_train[[column]].fillna(0))
        data_test['cluster'] = kmeans.predict(data_test[[column]].fillna(0))

        # Impute missing values within each cluster for the training set
        for cluster in range(kmeans.n_clusters):
            cluster_train_data = data_train[data_train['cluster'] == cluster]

            if cluster_train_data[column].isna().all():
                print(f"Cluster {cluster} in training set has all NaNs for column {column}.")
                continue
            
            # Impute missing values for the current cluster
            imputer = SimpleImputer(strategy=missing_values_strategy)
            imputed_train = imputer.fit_transform(cluster_train_data[[column]])

            # Update the imputed values back in the original DataFrame
            data_train.loc[data_train['cluster'] == cluster, column] = imputed_train

        # Impute for the test set
        for cluster in range(kmeans.n_clusters):
            cluster_test_data = data_test[data_test['cluster'] == cluster]

            if cluster_test_data[column].isna().all():
                print(f"Cluster {cluster} in testing set has all NaNs for column {column}.")
                continue
            
            # Use the same imputer fitted on the train cluster to impute test set
            imputed_test = imputer.transform(cluster_test_data[[column]])
            data_test.loc[data_test['cluster'] == cluster, column] = imputed_test

        # Drop the 'cluster' column after imputation
        data_train = data_train.drop(columns='cluster')
        data_test = data_test.drop(columns='cluster')

    return data_train, data_test


# def impute_missing_values(
#         data_train, 
#         data_test, 
#         columns_to_impute, 
#         missing_values_strategy="impute_mean",
#         ):
#     """
#     Impute missing values in a dataframe
#     """
#     data_train = data_train.copy()
#     data_test = data_test.copy()
#     if missing_values_strategy == "drop":
#         data_train[columns_to_impute] = data_train[columns_to_impute].dropna()
#         data_test[columns_to_impute] = data_test[columns_to_impute].dropna()
#     elif missing_values_strategy == "impute_zero":
#         data_train[columns_to_impute] = data_train[columns_to_impute].fillna(0)
#         data_test[columns_to_impute] = data_test[columns_to_impute].fillna(0)
#     elif missing_values_strategy == "impute_mean":
#         imputer = SimpleImputer(strategy="mean")
#         data_train[columns_to_impute] = imputer.fit_transform(data_train[columns_to_impute])
#         data_test[columns_to_impute] = imputer.transform(data_test[columns_to_impute])
#     return data_train, data_test

def select_features(features, mod):
    """
    Select features based on the modification type. Only keep the features corresponding to the list `mod`. 
    E.g. if mod="modc", only keep features related to modC, and discard those related to mC and hmC
    """
    if isinstance(mod, str):
        mod = [mod]
    return [f for f in features if any([m in f for m in mod]) or (("cpg_count" in f) or ("range" in f) or ("strand" in f))]

def split_train_test_data(data, features, target, missing_values_strategy="mean", use_SMOTE=False):
    """
    Split the data into training and testing sets using the specified test contig,
    impute missing values, and apply SMOTE if necessary.
    """
    print("Inside train-test split function")

    # Features to be used for imputation (excluding certain features like strand)
    impute_features = [f for f in features if not f.startswith("strand")]
    print(f"Features to impute: {len(impute_features)}, {impute_features}")
    
    # Initializing lists to store the indices and data for train and test
    all_train_indices = []
    all_test_indices = []
    synthetic_indices = []

    X_train_final = pd.DataFrame()
    y_train_final = pd.Series(dtype=float)
    X_test_final = pd.DataFrame()
    y_test_final = pd.Series(dtype=float)
    
    # Group data by contig (Chromosome)
    grouped = data.groupby('Chromosome')
    print(f"unique chromosome: {grouped.groups.keys()}")
    
    for contig, group in grouped:
        # 1. Split the data into train and test
        train, test = train_test_split(group, test_size=0.1, random_state=42)
        
        # Impute missing values within train and test sets
        train_imputed, test_imputed = impute_multimodal(train, test, impute_features, missing_values_strategy)

        # Check for NaNs
        # print(train_imputed.isna().any())
        # print(test_imputed.isna().any())
  
        if use_SMOTE:
            # Separate features and target in the train set
            X_train, y_train = train_imputed[features], train_imputed[target]
            # Remove any rows with missing target values
            y_train = y_train.dropna()
            X_train = X_train.loc[y_train.index]
            # Check if there's more than one class in the target
            if y_train.nunique() > 1:
                print(f"Contig {contig} - Before SMOTE: {X_train.shape}, {y_train.shape}")

                # Apply SMOTE to the training data
                smote = SMOTE(random_state=42)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

                # Track indices of synthetic samples
                synthetic_indices.extend(X_train_resampled.index.difference(X_train.index))

                # Append the resampled training data to the final training set
                X_train_final = pd.concat([X_train_final, X_train_resampled])
                y_train_final = pd.concat([y_train_final, y_train_resampled])
            else:
                print(f"Contig {contig} has only one class. Skipping SMOTE.")
                # Directly append the original train data
                X_train_final = pd.concat([X_train_final, X_train])
                y_train_final = pd.concat([y_train_final, y_train])

        else:
            # If not using classifier/SMOTE, simply add original train data
            X_train_final = pd.concat([X_train_final, train_imputed[features]])
            y_train_final = pd.concat([y_train_final, train_imputed[target]])

        # Append the test data to the final test set
        X_test_final = pd.concat([X_test_final, test_imputed[features]])
        y_test_final = pd.concat([y_test_final, test_imputed[target]])
        
        # Keep track of indices to remove from the original data
        all_train_indices.extend(train.index)
        all_test_indices.extend(test.index)

    ### debug print out
    # print(f"Contig {contig} split")
    # # Distribution of TPM in the test set
    # print(test_imputed[target].value_counts()/len(data))
    # print('-'*50)

    # Distribution of contigs in the train set (using indices)
    train_contig_distribution = data.loc[X_test_final.index, 'Chromosome'].value_counts() / data['Chromosome'].value_counts()
    print(train_contig_distribution)
    print('-'*50)
    ### debug print out

    print(f"Final train set: {X_train_final.shape}, {y_train_final.shape}")
    print(f"Final test set: {X_test_final.shape}, {y_test_final.shape}")

    return X_train_final, X_test_final, y_train_final, y_test_final


def stratified_sampling_per_contig(df,test_size=0.1,):
    
    # Group by contig (Chromosome)
    grouped = df.groupby('Chromosome')
    train_indices=[]
    
    for contig, group in grouped:

        # 1. Split the data without stratification
        train, test = train_test_split(
            group,
            test_size=test_size,
            random_state=42
        )
        train_indices.extend(train.index)
    train_set = df.loc[train_indices]
    
    # Exclude these indices to form the test set
    test_set = df.drop(train_indices)

    print(f"train set: {train_set.shape}")
    print(f"test set: {test_set.shape}")
    return train_set, test_set



def split_train_test_data_simple(data, features, target, multimodal=True):
    """
    Split the data into training and testing sets using the specified test contig
    """
    print("inside train test split")

    # Assuming 'features' is your list of feature names
    impute_features = [f for f in features if not f.startswith("strand")]
    print(f"features to impute:{len(impute_features)}, {impute_features}")

    data_train , data_test = stratified_sampling_per_contig(data)

    # Check the distribution of TPM in the test set
    print(data_test[target].value_counts()/len(data))
    print('-'*50)
    # Check the distribution of contigs in the test set
    print(data_test['Chromosome'].value_counts()/data['Chromosome'].value_counts())
    print('-'*50)

    print("before imputing")
    if multimodal:
        data_train, data_test = impute_multimodal(
            data_train,
            data_test,
            impute_features + [target],
            "mean"
        )
    else:
        "please import impute_missing_values"
        # data_train, data_test = impute_missing_values(
        #     data_train, 
        #     data_test, 
        #     impute_features + [target], 
        #     missing_values_strategy
        # )
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
        # transformed= np.log2(data+0.001)
        min_expressed=data[data!=0].min()
        print(f"minimum non-zero value: {min_expressed}, using {min_expressed/100} as a constant for log transform [plotting only]")
    
        print("y ranges from:")
        print(data.min())
        print(data.max())
        transformed= np.log2(data+min_expressed/100)

    return transformed



def preprocess(data, 
  features, 
  mod,
  target, 
  missing_values_strategy="mean", 
  multimodal=True,
  classifier=False,
  n_cat =0
):

  features = select_features(features, mod)
  print(features)


  data_cleaned = remove_obvious_outliers(data)
#   print(data_cleaned.head())

  # drop contig
  # Drop rows where 'Chromosome' is in the list ['X', 'Y']
  data_cleaned = data_cleaned[~data_cleaned['Chromosome'].isin(['X', 'Y'])]
  data_cleaned['Chromosome'] = data_cleaned['Chromosome'].cat.remove_unused_categories()

  print(f"removed chromosome X, Y: {data_cleaned['Chromosome'].unique()}")

#   if classifier:
#     labels = [k for k in range(n_cat)]
#     print(labels)
#     data_cleaned[target], binedges = pd.cut(
#         data_cleaned[target], bins=n_cat, labels=labels, retbins=True
#     )
#     print(f"bins: {binedges}")

#   X_train, X_test, y_train, y_test = split_train_test_data(
#         data_cleaned, 
#         features, 
#         target, 
#         missing_values_strategy,
#         use_SMOTE=False
#         )

  X_train, X_test, y_train, y_test = split_train_test_data_simple(
        data_cleaned, 
        features, 
        target, 
        )


  print("Shape of train data")
  X_train = scale_counts(X_train,"StandardScaler")
  print("Shape of test data")
  X_test = scale_counts(X_test, "StandardScaler")
  # X_train = log_transform(X_train)
  # X_test = log_transform(X_test)

  
  y_train = log_transform(y_train)
  y_test = log_transform(y_test)
  if classifier and n_cat>0:
    labels = [k for k in range(n_cat)]
    print(labels)
    y_train, train_binedges = pd.cut(
        y_train, bins=n_cat, labels=labels, retbins=True
    )
    y_test, binedges = pd.cut(
        y_test, bins=n_cat, labels=labels, retbins=True
    )
    print(f"bins: {train_binedges}")

  print(X_train.shape)
  print(X_test.shape)
  print(f"After train test split:")
  print(f"train target null count: {y_train.isna().sum()}")
  print(f"test target null count: {y_test.isna().sum()}")


  return X_train, X_test, y_train, y_test


######## END OF DATA PREPROCESSING ########




######## MODELLING ########

# regression
def tune_parameters(X_train, y_train, param_grid, model=xgb.XGBRegressor(eval_metric="rmsle"), scoring="r2"):
    """
    Identify the best hyperparameters of the XGBoost regressor using GridSearchCV
    """
    # kf = KFold(n_split=5, shuffle=True, random_state=42)
    
    search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=4).fit(X_train, y_train)
    return search.best_params_

# regression
def run_regressor(
        X_train, X_test, y_train, y_test,
        hyperparameters,
        random_state=1, 
        find_optimal_parameters=False,
        param_grid=None
        ):
    """
    Run the XGBoost regressor using the specified data and parameters
    """

    model = xgb.XGBRegressor(eval_metric="rmsle")
    scoring="r2"
  
    if find_optimal_parameters and param_grid is not None:
                best_params = tune_parameters(X_train, y_train, param_grid, model, scoring)
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
    param_grid=None
    ):
    """
    Run an xgboost classifier
    """
    print("in run_classifier:")
    print(y_train.value_counts())
    print(f"{y_train.isna().sum()/len(y_train)} is null")  # Count the number of NaN values

    # classifier = xgb.XGBClassifier(eval_metric=scoring)
    if len(y_test.unique())>2:

        print(f"multiclass: {len(y_test.unique())}")
        # scoring = "roc_auc_ovo"
        scoring = "f1_macro"
        classifier = xgb.XGBClassifier(eval_metric="logloss", objective="multi:softmax")
    else:
        scoring = "roc_auc"
        print(f"binary:  {len(y_test.unique())}")
        classifier = xgb.XGBClassifier(eval_metric="logloss")

    if find_optimal_parameters and param_grid is not None:
        print("grid searching...")
        best_params = tune_classifier_with_weights(X_train, y_train, param_grid, classifier, scoring )
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

    sample_weight = calculate_class_weight(y_train, n_categories=len(np.unique(y_train)))
    classifier.fit(X_train, y_train, sample_weight=sample_weight)
    # print("running classifier without adjusted weights")
    # classifier.fit(X_train, y_train)
    return classifier

def calculate_class_weight(y, n_categories):
    """ Calculate scale_pos_weight for XGBoost based on the class distribution. """
    
    class_weights = {}
    sample_weight = np.zeros_like(y, dtype=float)
    class_counts = np.bincount(y, minlength=n_categories)
    total_samples = len(y)
    # Calculate the class weights
    print("calculating class weights")
    for class_label in range(n_categories):
        weight = total_samples / (n_categories * class_counts[class_label]) if class_counts[class_label] > 0 else 0
        # class_weights[class_label] = weight
        print(f"{class_label}: {weight}")
        sample_weight[y == class_label] = weight

    return sample_weight

def tune_classifier_with_weights(X, y, param_grid, classifier, scoring, n_splits=5, random_state=42):
    """
    Tune XGBClassifier with class balancing and random StratifiedKFold
    """
    
    # StratifiedKFold ensures that each fold is randomized and maintains class proportions
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    print(f"skf defined with {n_splits} splits.")

    # GridSearchCV with custom fit parameters for each fold
    search = GridSearchCV(classifier, param_grid, cv=skf, scoring=scoring, n_jobs=4, refit=True)

    # Iterate through the StratifiedKFold splits
    for train_idx, val_idx in skf.split(X, y):
        print(type(y))
        y_train_fold = y.iloc[train_idx].values
        
        # Calculate class weights for the current fold
        sample_weight = calculate_class_weight(y_train_fold, n_categories=len(np.unique(y)))
        
        # Fit the model using the training data and the calculated sample weights
        search.fit(X.iloc[train_idx], y.iloc[train_idx], sample_weight=sample_weight)

    # Returning best parameters from the grid search
    return search.best_params_



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

def build_regression_model(X_train, X_test, y_train, y_test, target, default_hyperparameters, find_optimal_parameters=False, param_grid=None):
    
    model, df_metrics, y_test, y_pred = run_regressor(
        X_train, X_test, y_train, y_test,
        hyperparameters=default_hyperparameters,
        random_state=0,
        find_optimal_parameters=find_optimal_parameters,
        param_grid=param_grid
    )  

    print(df_metrics)
    plot_results(
        y_test, 
        y_pred, 
        f"Bulk RNA-seq - R^2={df_metrics.r2.values[0]:.2f}, Spearman R={df_metrics.spearman.values[0]:.2f},",
        target
    )


# def build_classification_model(X_train, X_test, y_train, y_test, default_hyperparameters, n_cat, find_optimal_parameters=False, param_grid=None):
#     # create n categories based on the response values
#     labels = [k for k in range(n_cat)]
#     # df_features_expression["category"] = pd.cut(
#     #     df_features_expression["Response"], bins=n, labels=labels
#     # )
#     print(labels)
#     y_train_binned, binedges = pd.cut(
#         y_train, bins=n_cat, labels=labels, retbins=True
#     )
#     print(y_train_binned.value_counts(), binedges)
#     y_test_binned, binedges = pd.cut(
#         y_test, bins=n_cat, labels=labels,retbins=True
#     )
def build_classification_model(X_train, X_test, y_train_binned, y_test_binned, default_hyperparameters, n_cat, find_optimal_parameters=False, param_grid=None):
    
    # run the classifier
    c, df_metrics = run_classifier(
        X_train, X_test, y_train_binned, y_test_binned,
        hyperparameters=default_hyperparameters,
        find_optimal_parameters=find_optimal_parameters,
        random_state=1,
        param_grid=param_grid
        )
    print(f"for {n_cat} categories, acc metrics:")
    print(df_metrics)
    return c, df_metrics



######## END oF UTILITY FUNCTIONS ########

def run_main(input, rna_data,modifications, target, expressed_only):
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
    
    # expression data
    if target == "TPM":
        zscore_threshold = 0.8
        multimodal = True
        df_features_expression = load_rna_expression_tpm(rna_data, df_features, transcripts, expressed_only)
        print("TPM expression data loaded")
    elif target == "rpkm":
        selected_transcripts = transcripts.groupby('gene_id').apply(
            select_transcript_based_on_tag
        ).reset_index(drop=True)
        print(f"transcript downloaded, selected transcripts: {selected_transcripts}")
   
        zscore_threshold = 1
        multimodal = False
        df_expression = load_rna_expression_rpm(rna_data, selected_transcripts, transcripts, target)
        print(f"rpm expression data loaded: {df_expression.shape}")
        print(f"null counts:{df_expression.isnull().sum()}")

        df_features_expression = pd.merge(
            df_features,
            df_expression,
            on=["Chromosome", "Start", "End"],
            how="inner",
        )
    
    # print(f"features and target merged: {df_features_expression}")
    print(f"shape: {df_features_expression.shape}")
    print('='*200)
    
    
    # preprocessing:
    orig_feature_shape = df_features_expression.shape
    # drop columns with Nan
    df_features_expression_clean = df_features_expression.dropna(axis=1, how='all')
    # frop contig column
    df_features_expression_clean = df_features_expression_clean.drop('contig', axis=1)
    print(f"Nan features(columns) dropped: {orig_feature_shape[1]} --> {df_features_expression_clean.shape[1]}")

    print(f"selected target unit: {target}")
    print("PREPROCESSING...")

    
    print(f"removing rows with {target} zscore > {zscore_threshold} - target outliers")
    df_clean = remove_target_outliers(df_features_expression_clean, target, zscore_threshold)
    plot_target_kde(df_clean,target)
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

    default_hyperparameters = {
        'n_estimators': 600, 
        'colsample_bytree': 0.8, 
        'eta': 0.01, 
        'max_depth': 5, 
        'min_child_weight': 10, 
        'subsample': 0.8
    }

    # Building a regressor:

    # train-test split
    X_train, X_test, y_train, y_test = preprocess(
    data=df_clean,
    features=features,
    mod = modifications,
    target=target,
    missing_values_strategy="mean",
    multimodal = multimodal,
    classifier = False
    )
    print('='*200)


    regression_gridsearch = False
    if regression_gridsearch:
        param_grid = {
            "max_depth": [5, 6, 7],
            "n_estimators": [600, 800, 1000], #[200, 300, 400, 500, 600, 700, 800],
            "subsample": [0.6, 0.7, 0.8],
            "min_child_weight": [1, 5, 10],
            "colsample_bytree": [0.8, 0.9],#[0.75, 0.8, 0.85, 0.9],
            "eta": [0.01, 0.02, 0.03]#[0.01, 0.02, 0.03, 0.04, 0.05],
        }
        print("grid serching (Regression)")
    else:
        param_grid = None
        print(f"training with {default_hyperparameters}")

    build_regression_model(X_train, X_test, y_train, y_test, target, default_hyperparameters, regression_gridsearch, param_grid) #skip gridsearch for now
    
    
    print('='*200)

    # Build a classifier:
    classifier_gridsearch = True
    
    number_of_categories = np.arange(2, 5, 1)
    classifiers = {}
    classifier_res = pd.DataFrame(columns=["number_of_categories", "accuracy", "macro_f1", "auc"])

    # Loop through categories
    for n in number_of_categories:
        # train-test split
        X_train, X_test, y_train, y_test = preprocess(
        data=df_clean,
        features=features,
        mod = modifications,
        target=target,
        missing_values_strategy="mean",
        multimodal = multimodal,
        classifier = True,
        n_cat=n
        )
        
        if classifier_gridsearch:
            param_grid = {
            "max_depth": [5, 6, 7],
            "n_estimators": [300, 400, 500, 600, 700, 800],
            "subsample": [0.6, 0.7, 0.8],
            "min_child_weight": [1, 5, 10],
            "colsample_bytree": [0.75, 0.8, 0.85, 0.9],
            "eta": [0.01, 0.02, 0.03, 0.04, 0.05],
            }
            print(f"training {n} category model with grid serching (classification)")
        else:
            param_grid = None
            print(f"training {n} category model with {default_hyperparameters}")
        
        classifier, df_metrics = build_classification_model(X_train, X_test, y_train, y_test, default_hyperparameters, n, classifier_gridsearch, param_grid)
        
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




