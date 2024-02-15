import os
import numpy as np
import pandas as pd
import tensorflow as tf

def create_X_hein_daily(author_info: pd.DataFrame, covariates: str, ideal_topic_dim: int):
    """ Creates the model matrix X for the regression of ideological positions for the hein-daily dataset.
    Also initial ideological positions are set.
    
    Args:
        author_info: The dataframe containing the author-level covariates.
        covariates: A string that defines what should the model matrix contain.
        ideal_topic_dim: Dimension of the ideological positions.
            Number of topics if topic-specific ideological positions.
            Or 1 if fixed ideological positions for all authors.

    Returns:
        X: Model matrix for the regression of ideological positions constructed using author-specific covariates.
        initial_ideal_location: Initial value for the ideological positions.
            float[num_authors, ideal_topic_dim]

    """
    num_authors = author_info.shape[0]
    if covariates == "None":
        X = np.array([np.ones(num_authors)])  # intercept term,
    elif covariates == "party":
        X = np.array([np.ones(num_authors),  # intercept term,
                      ## Democratic senators ('D') are is baseline category
                      1.0 * (author_info["party"] == "R"),  # indicator of Republican senator
                      1.0 * (author_info["party"] == "I")])  # indicator of Independent senator
    elif covariates == "all_no_int":
        religion_dummies = author_info['RELIGION'].str.get_dummies()
        X = np.array([np.ones(num_authors),  # intercept term,
                      ## Democratic senators ('D') are is baseline category
                      1.0 * (author_info["party"] == "R"),  # indicator of Republican senator
                      1.0 * (author_info["party"] == "I"),  # indicator of Independent senator
                      ## Males ('M') are baseline category.
                      1.0 * (author_info["gender"] == "F"),  # indicator of a female senator
                      ## Northeast ('Northeast') region is baseline category.
                      1.0 * (author_info["region"] == "Midwest"),  # indicator of a Midwest senator
                      1.0 * (author_info["region"] == "Southeast"),  # indicator of a Southeast senator
                      1.0 * (author_info["region"] == "South"),  # indicator of a South senator
                      1.0 * (author_info["region"] == "West"),  # indicator of a West senator
                      ## Age of the senator - linear parametrization
                      # author_info['age'],
                      ## Age of the senator - generation indicators
                      # Silent     22   (1928-1945) - very old
                      # Boomers    62   (1946-1964) - old
                      # Gen X      15   (1965-1980) - young
                      1.0 * (author_info['generation'] == 'Boomers'),
                      1.0 * (author_info['generation'] == 'Gen X'),
                      ## Experience - cumulative number of congress as a senator / member of congress
                      # 1.0 * (author_info['exper_cong'] == '(10, 100]')  # Long experience in Congress - default
                      1.0 * (author_info['exper_cong'] == '(1, 10]'),  # already in Congress
                      1.0 * (author_info['exper_cong'] == '(0, 1]'),  # in Congress for the first time
                      # 1.0 * (author_info['exper_chamber'] == '(1, 5]')  # already in Senate
                      # 1.0 * (author_info['exper_chamber'] == '(5, 100]')  # Long experience in Senate
                      # OR USE 'FRESHMAN.' which is Yes/No on whether it is senator's first session.
                      ## RELIGION
                      # Catholic                          25
                      # Presbyterian                      14
                      # Baptist                           10
                      # Jewish                             9
                      # Unspecified/Other (Protestant)     9
                      # Methodist                          9
                      # Lutheran                           7
                      # Mormon                             7
                      # Anglican/Episcopal                 4
                      # Don’t Know/Refused                 2
                      # Congregationalist                  1
                      # Nondenominational Christian        1
                      # Buddhist                           1
                      # Catholic is the baseline
                      # OR use Other group as intercept (frequencies < 7)
                      religion_dummies['Catholic'].to_numpy(),
                      religion_dummies['Presbyterian'].to_numpy(),
                      religion_dummies['Baptist'].to_numpy(),
                      religion_dummies['Jewish'].to_numpy(),
                      # religion_dummies['Unspecified/Other (Protestant)'].to_numpy(),
                      religion_dummies['Methodist'].to_numpy(),
                      religion_dummies['Lutheran'].to_numpy(),
                      religion_dummies['Mormon'].to_numpy(),
                      ])
    elif covariates == "all":
        religion_dummies = author_info['RELIGION'].str.get_dummies()
        X = np.array([np.ones(num_authors),  # intercept term,
                      ## Democratic senators ('D') are is baseline category
                      1.0 * (author_info["party"] == "R"),  # indicator of Republican senator
                      1.0 * (author_info["party"] == "I"),  # indicator of Independent senator
                      ## Males ('M') are baseline category.
                      1.0 * (author_info["gender"] == "F"),  # indicator of a female senator
                      ## Northeast ('Northeast') region is baseline category.
                      1.0 * (author_info["region"] == "Midwest"),  # indicator of a Midwest senator
                      1.0 * (author_info["region"] == "Southeast"),  # indicator of a Southeast senator
                      1.0 * (author_info["region"] == "South"),  # indicator of a South senator
                      1.0 * (author_info["region"] == "West"),  # indicator of a West senator
                      ## Age of the senator - linear parametrization
                      # author_info['age'],
                      ## Age of the senator - generation indicators
                      # Silent     22   (1928-1945) - very old
                      # Boomers    62   (1946-1964) - old
                      # Gen X      15   (1965-1980) - young
                      1.0 * (author_info['generation'] == 'Boomers'),
                      1.0 * (author_info['generation'] == 'Gen X'),
                      ## Experience - cumulative number of congress as a senator / member of congress
                      # 1.0 * (author_info['exper_cong'] == '(10, 100]')  # Long experience in Congress - default
                      1.0 * (author_info['exper_cong'] == '(1, 10]'),  # already in Congress
                      1.0 * (author_info['exper_cong'] == '(0, 1]'),  # in Congress for the first time
                      # 1.0 * (author_info['exper_chamber'] == '(1, 5]')  # already in Senate
                      # 1.0 * (author_info['exper_chamber'] == '(5, 100]')  # Long experience in Senate
                      # OR USE 'FRESHMAN.' which is Yes/No on whether it is senator's first session.
                      ## RELIGION
                      # Catholic                          25
                      # Presbyterian                      14
                      # Baptist                           10
                      # Jewish                             9
                      # Unspecified/Other (Protestant)     9
                      # Methodist                          9
                      # Lutheran                           7
                      # Mormon                             7
                      # Anglican/Episcopal                 4
                      # Don’t Know/Refused                 2
                      # Congregationalist                  1
                      # Nondenominational Christian        1
                      # Buddhist                           1
                      # Catholic is the baseline
                      # OR use Other group as intercept (frequencies < 7)
                      religion_dummies['Catholic'].to_numpy(),
                      religion_dummies['Presbyterian'].to_numpy(),
                      religion_dummies['Baptist'].to_numpy(),
                      religion_dummies['Jewish'].to_numpy(),
                      # religion_dummies['Unspecified/Other (Protestant)'].to_numpy(),
                      religion_dummies['Methodist'].to_numpy(),
                      religion_dummies['Lutheran'].to_numpy(),
                      religion_dummies['Mormon'].to_numpy(),
                      # religion_dummies['Lutheran'].to_numpy(),
                      # religion_dummies['Presbyterian'].to_numpy(),
                      # 1.0 * (religion_dummies['Congregationalist'] + religion_dummies['Baptist'] + religion_dummies[
                      #     'Anglican/Episcopal'] + religion_dummies['Methodist'] > 0),
                      # religion_dummies['Jewish'].to_numpy(),
                      # religion_dummies['Mormon'].to_numpy(),
                      # 1.0 * (religion_dummies['Unspecified/Other (Protestant)'] + religion_dummies['Don’t Know/Refused'] +
                      #        religion_dummies['Nondenominational Christian'] > 0),
                      # religion_dummies['Buddhist'].to_numpy(),
                      # 1.0 * (author_info['RELIGION'] == 'Lutheran'),
                      # 1.0 * (author_info['RELIGION'] == 'Presbyterian'),
                      # 1.0 * (author_info['RELIGION'] in ['Congregationalist', 'Baptist', 'Anglican/Episcopal', 'Methodist']),
                      # 1.0 * (author_info['RELIGION'] == 'Jewish'),
                      # 1.0 * (author_info['RELIGION'] == 'Mormon'),
                      # 1.0 * (author_info['RELIGION'] in ['Unspecified/Other (Protestant)', 'Don’t Know/Refused', 'Nondenominational Christian']),
                      # 1.0 * (author_info['RELIGION'] == 'Buddhist'),
                      ### Interaction terms with political party
                      ## Republican
                      # Gender
                      1.0 * (author_info["party"] == "R") * (author_info["gender"] == "F"),
                      # Region
                      1.0 * (author_info["party"] == "R") * (author_info["region"] == "Midwest"),
                      1.0 * (author_info["party"] == "R") * (author_info["region"] == "Southeast"),
                      1.0 * (author_info["party"] == "R") * (author_info["region"] == "South"),
                      1.0 * (author_info["party"] == "R") * (author_info["region"] == "West"),
                      # Generation
                      1.0 * (author_info["party"] == "R") * (author_info['generation'] == 'Boomers'),
                      1.0 * (author_info["party"] == "R") * (author_info['generation'] == 'Gen X'),
                      # Experience
                      1.0 * (author_info["party"] == "R") * (author_info['exper_cong'] == '(1, 10]'),
                      1.0 * (author_info["party"] == "R") * (author_info['exper_cong'] == '(0, 1]'),
                      # Religion
                      1.0 * (author_info["party"] == "R") * religion_dummies['Catholic'].to_numpy(),
                      1.0 * (author_info["party"] == "R") * religion_dummies['Presbyterian'].to_numpy(),
                      1.0 * (author_info["party"] == "R") * religion_dummies['Baptist'].to_numpy(),
                      1.0 * (author_info["party"] == "R") * religion_dummies['Jewish'].to_numpy(),
                      # 1.0 * (author_info["party"] == "R") * religion_dummies['Unspecified/Other (Protestant)'].to_numpy(),
                      1.0 * (author_info["party"] == "R") * religion_dummies['Methodist'].to_numpy(),
                      1.0 * (author_info["party"] == "R") * religion_dummies['Lutheran'].to_numpy(),
                      1.0 * (author_info["party"] == "R") * religion_dummies['Mormon'].to_numpy(),
                      ## Independent
                      ## There are only 2 Independent senators --> very likely that the corresponding interaction term
                      ## creates a column that consists of zeros only --> X would not be of full rank.
                      ## However, the prior distribution for the coefficients should help us avoid errors.
                      # Gender
                      1.0 * (author_info["party"] == "I") * (author_info["gender"] == "F"),
                      # Region
                      1.0 * (author_info["party"] == "I") * (author_info["region"] == "Midwest"),
                      1.0 * (author_info["party"] == "I") * (author_info["region"] == "Southeast"),
                      1.0 * (author_info["party"] == "I") * (author_info["region"] == "South"),
                      1.0 * (author_info["party"] == "I") * (author_info["region"] == "West"),
                      # Generation
                      1.0 * (author_info["party"] == "I") * (author_info['generation'] == 'Boomers'),
                      1.0 * (author_info["party"] == "I") * (author_info['generation'] == 'Gen X'),
                      # Experience
                      1.0 * (author_info["party"] == "I") * (author_info['exper_cong'] == '(1, 10]'),
                      1.0 * (author_info["party"] == "I") * (author_info['exper_cong'] == '(0, 1]'),
                      # Religion
                      1.0 * (author_info["party"] == "I") * religion_dummies['Catholic'].to_numpy(),
                      1.0 * (author_info["party"] == "I") * religion_dummies['Presbyterian'].to_numpy(),
                      1.0 * (author_info["party"] == "I") * religion_dummies['Baptist'].to_numpy(),
                      1.0 * (author_info["party"] == "I") * religion_dummies['Jewish'].to_numpy(),
                      # 1.0 * (author_info["party"] == "I") *religion_dummies['Unspecified/Other (Protestant)'].to_numpy(),
                      1.0 * (author_info["party"] == "I") * religion_dummies['Methodist'].to_numpy(),
                      1.0 * (author_info["party"] == "I") * religion_dummies['Lutheran'].to_numpy(),
                      1.0 * (author_info["party"] == "I") * religion_dummies['Mormon'].to_numpy(),
                      ])
    else:
        raise ValueError('Unrecognized choice of covariates for the author-level regression on ideal points.')
    X = X.transpose()
    X = tf.cast(tf.constant(X), tf.float32)

    # Initial locations: -1 for Democrats, 1 for Republicans, 0 for others
    auxloc = -X[:, 0] + 2*(author_info["party"] == "R") + 1*(author_info["party"] == "I")
    initial_ideal_location = tf.repeat(auxloc[:, tf.newaxis], ideal_topic_dim, axis=1)
    
    return X, initial_ideal_location


def create_X_cze_senate(author_info: pd.DataFrame, covariates: str, ideal_topic_dim: int):
    """ Creates the model matrix X for the regression of ideological positions for the cze_senate senate dataset.
    Also initial ideological positions are set.

    Args:
        author_info: The dataframe containing the author-level covariates.
        covariates: A string that defines what should the model matrix contain.
        ideal_topic_dim: Dimension of the ideological positions.
            Number of topics if topic-specific ideological positions.
            Or 1 if fixed ideological positions for all authors.

    Returns:
        X: Model matrix for the regression of ideological positions constructed using author-specific covariates.
        initial_ideal_location: Initial value for the ideological positions.
            float[num_authors, ideal_topic_dim]

    """
    num_authors = author_info.shape[0]
    if covariates == "None":
        X = np.array([np.ones(num_authors)])  # intercept term,
    elif covariates == "party":
        X = np.array([np.ones(num_authors),  # intercept term,
                      ## baseline category are other remaining parties
                      1.0 * (author_info["party"] == "ODS"),  # indicator of ODS senator
                      1.0 * (author_info["party"] == "CSSD"),  # indicator of ČSSD senator
                      1.0 * (author_info["party"] == "ANO"),  # indicator of ANO senator
                      1.0 * (author_info["party"] == "TOP09"),  # indicator of TOP-09 senator
                      1.0 * (author_info["party"] == "STAN"),  # indicator of STAN senator
                      1.0 * (author_info["party"] == "KDU-CSL"),  # indicator of KDU-ČSL senator
                      1.0 * (author_info["party"] == "KSCM"),  # indicator of KSČM senator
                      1.0 * (author_info["party"] == "NK")
        ])
    else:
        raise ValueError('Unrecognized choice of covariates for the author-level regression on ideal points.')
    X = X.transpose()
    X = tf.cast(tf.constant(X), tf.float32)
    print("Regression matrix X: ")
    print(X)
    print("X summed over first dimension: ")
    print(tf.reduce_sum(X, axis=0))
    print("X summed over second dimension: ")
    print(tf.reduce_sum(X, axis=1))

    # Initial locations: 1 for ODS, -1 for ČSSD, 0 for others
    auxloc = 1.0 * (author_info["party"] == "ODS") - 1.0 * (author_info["party"] == "CSSD")
    auxloc = tf.cast(tf.constant(auxloc), tf.float32)
    print(auxloc)
    initial_ideal_location = tf.repeat(auxloc[:, tf.newaxis], ideal_topic_dim, axis=1)

    return X, initial_ideal_location

def create_X_pharma(author_info: pd.DataFrame, covariates: str, ideal_topic_dim: int):
    """ Creates the model matrix X for the regression of ideological positions for the pharma dataset.
    Also initial ideological positions are set.

    Args:
        author_info: The dataframe containing the author-level covariates.
        covariates: A string that defines what should the model matrix contain.
        ideal_topic_dim: Dimension of the ideological positions.
            Number of topics if topic-specific ideological positions.
            Or 1 if fixed ideological positions for all authors.

    Returns:
        X: Model matrix for the regression of ideological positions constructed using author-specific covariates.
        initial_ideal_location: Initial value for the ideological positions.
            float[num_authors, ideal_topic_dim]

    """
    num_authors = author_info.shape[0]
    if covariates == "None":
        X = np.array([np.ones(num_authors)])  # intercept term,
        X = X.transpose()
    elif covariates == "Nutzen":
        X = np.array([np.ones(num_authors),  # intercept term,
                      author_info["Nutzen"]])  # 0/1 variable
        X = X.transpose()
    elif covariates == "additive":
        company_dummies = author_info['Company'].str.get_dummies()
        # Highest amount of documents is by 'Novartis Pharma GmbH'
        company_dummies_no_baseline = company_dummies.drop('Novartis Pharma GmbH', axis=1)
        an_dummies = author_info['Anwendungsgebiet.general'].str.get_dummies()
        # Highest frequency: 'onkologische Erkrankungen'
        an_dummies_no_baseline = an_dummies.drop('onkologische Erkrankungen', axis=1)

        X = np.array([np.ones(num_authors),  # intercept term,
                      author_info["Nutzen"]])  # 0/1 variable
        X = X.transpose()
        X = tf.concat([X, company_dummies_no_baseline.to_numpy(), an_dummies_no_baseline.to_numpy()], 1)

    else:
        raise ValueError('Unrecognized choice of covariates for the author-level regression on ideal points.')
    X = tf.cast(tf.constant(X), tf.float32)

    # Initial locations: zeros for all documents
    auxloc = tf.zeros(num_authors)
    initial_ideal_location = tf.repeat(auxloc[:, tf.newaxis], ideal_topic_dim, axis=1)

    return X, initial_ideal_location

def create_X_fomc(author_info: pd.DataFrame, covariates: str, ideal_topic_dim: int):
    """ Creates the model matrix X for the regression of ideological positions for the pharma dataset.
    Also initial ideological positions are set.

    Args:
        author_info: The dataframe containing the author-level covariates.
        covariates: A string that defines what should the model matrix contain.
        ideal_topic_dim: Dimension of the ideological positions.
            Number of topics if topic-specific ideological positions.
            Or 1 if fixed ideological positions for all authors.

    Returns:
        X: Model matrix for the regression of ideological positions constructed using author-specific covariates.
        initial_ideal_location: Initial value for the ideological positions.
            float[num_authors, ideal_topic_dim]

    """
    num_authors = author_info.shape[0]
    title_dummies = author_info['title'].str.get_dummies()
    year_dummies = author_info['year'].str.get_dummies()
    length_dummies = author_info['flength'].str.get_dummies()
    laughter_dummies = author_info['flaughter'].str.get_dummies()

    if covariates == "None":
        X = np.array([np.ones(num_authors)])  # intercept term,
    elif covariates == "gender":
        X = np.array([np.ones(num_authors),  # intercept term,
                      author_info["gender"]])  # 0/1 variable
    elif covariates == "gender+title+year+flength+flaughter":
        X = np.array([np.ones(num_authors),  # intercept term,
                      # speaker-specific covariates
                      author_info["gender"],
                      title_dummies['chairman'].to_numpy(),
                      title_dummies['vice chairman'].to_numpy(),
                      # meeting-specific covariates
                      year_dummies['2006'].to_numpy(),
                      year_dummies['2007'].to_numpy(),
                      length_dummies['medium'].to_numpy(),
                      length_dummies['long'].to_numpy(),
                      laughter_dummies['medium'].to_numpy(),
                      laughter_dummies['relaxed'].to_numpy()
                     ])
    else:
        raise ValueError('Unrecognized choice of covariates for the author-level regression on ideal points.')
    X = X.transpose()
    X = tf.cast(tf.constant(X), tf.float32)

    # Initial locations: -1 for chairmans, 0 vice chairmans, 1 for ms_or_mr
    auxloc = -title_dummies['chairman'].to_numpy() + title_dummies['ms_or_mr'].to_numpy()
    auxloc = tf.cast(tf.constant(auxloc), tf.float32)
    initial_ideal_location = tf.repeat(auxloc[:, tf.newaxis], ideal_topic_dim, axis=1)

    return X, initial_ideal_location

def create_X(data_name: str, author_info: pd.DataFrame, covariates: str, ideal_topic_dim: int):
    """ Triggers the function that computes regression matrix X and initial values for ideological positions.
    The triggered function is different for different datasets.
    If you are working with a new dataset you have to implement your own 'create_X_...' function.

    Args:
        data_name: Name of the dataset to determine X.
        author_info: The dataframe containing the author-level covariates.
        covariates: A string that defines what should the model matrix contain.
        ideal_topic_dim: Dimension of the ideological positions.
            Number of topics if topic-specific ideological positions.
            Or 1 if fixed ideological positions for all authors.

    Returns:
        X: Model matrix for the regression of ideological positions constructed using author-specific covariates.
        initial_ideal_location: Initial value for the ideological positions.
            float[num_authors, ideal_topic_dim]

    """
    if data_name == "hein-daily":
        X, initial_ideal_location = create_X_hein_daily(author_info, covariates, ideal_topic_dim)
    elif data_name == 'cze_senate':
        X, initial_ideal_location = create_X_cze_senate(author_info, covariates, ideal_topic_dim)
    elif data_name == 'pharma':
        X, initial_ideal_location = create_X_pharma(author_info, covariates, ideal_topic_dim)
    elif data_name == 'fomc':
        X, initial_ideal_location = create_X_fomc(author_info, covariates, ideal_topic_dim)
    else:
        raise ValueError("Procedure for creating model matrix X and initial ideological positions for dataset " + data_name + " is not implemented within 'create_X.py'.")

    return X, initial_ideal_location