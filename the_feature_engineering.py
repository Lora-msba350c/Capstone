import pandas as pd
import numpy as np

def apply_feature_engineering(df_clean):
    # 1. Basic Derived Features
    df_clean['WEIGHT_PER_TEU'] = df_clean['TOTAL_WEIGHT'] / df_clean['TOTAL_TEUS']
    df_clean['ROUTE'] = df_clean['POINT_LOAD'] + '-' + df_clean['POINT_DISCH']
    df_clean['MONTH_YEAR'] = df_clean['YEAR'].astype(str) + '-' + df_clean['MONTH'].astype(str).str.zfill(2)
    df_clean['MONTH_INT'] = df_clean['YEAR'] * 12 + df_clean['MONTH']

    # 2. Rare Route Flag
    route_counts = df_clean['ROUTE'].value_counts()
    rare_routes = set(route_counts[route_counts < 10].index)
    df_clean['RARE_ROUTE'] = df_clean['ROUTE'].apply(lambda x: 1 if x in rare_routes else 0)

    # 3. Monthly TEU Deviation
    monthly_teu_avg = df_clean.groupby('MONTH_YEAR')['TOTAL_TEUS'].transform('mean')
    df_clean['TEU_DEVIATION'] = df_clean['TOTAL_TEUS'] / monthly_teu_avg

    # 4. Commodity Weight Ratio
    median_weight_per_teu = df_clean.groupby('COMMODITY_CODE')['WEIGHT_PER_TEU'].transform('median')
    df_clean['COMMODITY_WEIGHT_RATIO'] = df_clean['WEIGHT_PER_TEU'] / median_weight_per_teu

    # 5. Unusual Container + Commodity Combo
    combo_counts = df_clean.groupby(['CONTAINER_TYPE', 'COMMODITY_CATEGORY']).size().reset_index(name='COUNT')
    rare_combos = set(combo_counts[combo_counts['COUNT'] < 100].apply(lambda r: (r['CONTAINER_TYPE'], r['COMMODITY_CATEGORY']), axis=1))
    df_clean['UNUSUAL_CONTAINER_COMMODITY'] = df_clean.apply(
        lambda row: 1 if (row['CONTAINER_TYPE'], row['COMMODITY_CATEGORY']) in rare_combos else 0, axis=1
    )

    # 6. Partner TEU Deviation
    partner_avg_teu = df_clean.groupby('PARTNER_CODE')['TOTAL_TEUS'].transform('mean')
    df_clean['PARTNER_TEU_DEVIATION'] = df_clean['TOTAL_TEUS'] / partner_avg_teu

    # 7. Is New Route for Partner
    partner_route_count = df_clean.groupby(['PARTNER_CODE', 'ROUTE'])['JOB_REFERENCE'].transform('count')
    df_clean['ROUTE_IS_NEW_FOR_PARTNER'] = (partner_route_count == 1).astype(int)

    # 8. Partner Activity Span
    partner_activity_span = df_clean.groupby('PARTNER_CODE')['MONTH_INT'].transform(lambda x: x.max() - x.min() + 1)
    df_clean['PARTNER_ACTIVITY_SPAN'] = partner_activity_span

    # 9. HS Code Length
    df_clean['HS_CODE_LENGTH'] = df_clean['COMMODITY_CODE'].astype(str).str.len()

    # 10. Seasonal Peak Flag
    peak_months = {
        'Chemicals & Allied Industries': [11, 12],
        'Foodstuffs': [11, 12],
        'Footwear / Headgear': [9],
        'Machinery / Electrical': [11, 12],
        'Metals': [11, 12],
        'Mineral Products': [11],
        'Miscellaneous': [11, 12],
        'Personal Property': [8, 11, 12],
        'Plastics / Rubbers': [11, 12],
        'Raw Hides, Skins, Leather, & Furs': [6],
        'Stone / Glass': [11, 12],
        'Textiles': [11],
        'Transportation': [4, 11, 12],
        'Vegetable Products': [11, 12]
    }
    df_clean['SEASONAL_PEAK_FLAG'] = df_clean.apply(
        lambda row: 1 if row['MONTH'] in peak_months.get(row['COMMODITY_CATEGORY'], []) else 0,
        axis=1
    )

    return df_clean
