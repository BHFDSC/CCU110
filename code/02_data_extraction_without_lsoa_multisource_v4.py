# Databricks notebook source
# MAGIC %md
# MAGIC # 0. Setup

# COMMAND ----------

import pyspark.sql.functions as f
import pyspark.sql.types as t
from pyspark.sql import Window

from functools import reduce

# import databricks.koalas as ks
import pandas as pd
import numpy as np

import re
import io
import os
import datetime

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import seaborn as sns

print("Matplotlib version: ", matplotlib.__version__)
print("Seaborn version: ", sns.__version__)
_datetimenow = datetime.datetime.now() # .strftime("%Y%m%d")
print(f"_datetimenow:  {_datetimenow}")

# COMMAND ----------

# MAGIC %run "/Shared/SHDS/common/functions"

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Parameters

# COMMAND ----------

# MAGIC %run "./01_parameters"

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Data

# COMMAND ----------

hes_ae_all_years_archive = spark.table(f'{dbc}.hes_ae_all_years_archive')
hes_apc_all_years_archive = spark.table(f'{dbc}.hes_apc_all_years_archive')
demographics = spark.table(f'{dsa}.hds_curated_assets__demographics_2026_01_20')
token_pseudo_id_lookup_archive = spark.table(f'{dbc}.token_pseudo_id_lookup_archive')
lsoa_multisource = spark.table(f'{dsa}.hds_curated_assets__lsoa_multisource_2026_01_20')
ecds = spark.table(f'{dbc}.lowlat_ecds_all_years_archive')

# lookups
lsoa_2011 = spark.table(f'{dss}.ons_lsoa_2011')
imd_lookup = spark.table(f'{dsa}.hds_cur_lsoa_2011_imd_lookup')

codelist_path = f'/Workspace{os.path.dirname(os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()))}/CCU110_01/lsoa_2011_icb_lookup'
lsoa_2011_icb = pd.read_csv(codelist_path, keep_default_na=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Prepare

# COMMAND ----------

# DBTITLE 1,HES A&E
# Take latest batch of HES A&E which also provides high coverage until December 2023 (archived_on == '2025-02-04')
# Select columns of interest
# Filter to study period of interest
# Filter based on age 
hes_ae_all_years_filtered = (
    hes_ae_all_years_archive
    .filter(f.col('archived_on') == '2025-02-04')
    .select('person_id_deid',
    'aekey',
    'aearrivalmode',
    'arrivaldate',
    'arrivaltime',
    'FYEAR',
    'LSOA11',
    'PROCODE3',
    'PROCODE5')
    .filter((f.col('arrivaldate') >= f.to_date(f.lit('2013-01-01'))) & (f.col('arrivaldate') <= f.to_date(f.lit('2023-12-31'))))
)

display(hes_ae_all_years_filtered)

# COMMAND ----------

# Flag records with non-England LSOAs
hes_ae_all_years_filtered = (
    hes_ae_all_years_filtered
    .withColumn('lsoa_in_england', f.when(f.substring(f.col('LSOA11'), 1, 1) == 'E', 1)
                .when((f.col('LSOA11').isNotNull()) & (f.substring(f.col('LSOA11'), 1, 1) != 'E'), 0)
                .otherwise(None))
)

tab(hes_ae_all_years_filtered, 'lsoa_in_england')

# COMMAND ----------

# DBTITLE 1,Pseudo id table
token_pseudo_id_lookup_filtered = (
    token_pseudo_id_lookup_archive
    .filter(f.col('archived_on') == '2026-01-20')
    .withColumn('id_type', f.when(f.col('mps_id') == True, 'MPS ID')
                .when(f.col('single_use_id') == True, 'One-time-use ID')
                .when(f.col('valid_nhs_number') == True, 'NHS Number')
                .otherwise('None'))
    .join(hes_ae_all_years_filtered.select(f.col('person_id_deid').alias('pseudo_id')).distinct(), 'pseudo_id', how='inner')
)

display(token_pseudo_id_lookup_filtered)

# COMMAND ----------

# DBTITLE 1,Demographics
# Filter to HES A&E table
demographics_filtered = (
    demographics
    .select('person_id', 'date_of_birth', 'date_of_death', 'sex_code', 'ethnicity_18_group', 'lsoa', 'imd_quintile')
    .join(hes_ae_all_years_filtered.withColumnRenamed('person_id_deid', 'person_id').select('person_id').distinct(), 'person_id', how='inner')
    .join(token_pseudo_id_lookup_filtered.select(f.col('pseudo_id').alias('person_id'), 'id_type'), 'person_id', how='left') # add id_type column
    .withColumn('in_demo', f.lit(1))
)

display(demographics_filtered)

# COMMAND ----------

# DBTITLE 1,Filter based on age
hes_ae_all_years_filtered_age = (
    hes_ae_all_years_filtered
    .join(demographics_filtered.select(f.col('person_id').alias('person_id_deid'), 'date_of_birth').distinct(), 'person_id_deid', how='left')
    .withColumn('age_at_arrival_date', f.floor(f.date_diff(f.col('arrivaldate'), f.col('date_of_birth'))/365.25))
    .filter(f.col('age_at_arrival_date') >= 18)
    .drop('date_of_birth')
)

display(hes_ae_all_years_filtered_age)

# COMMAND ----------

demographics_filtered_age = (
    demographics_filtered
    .join(hes_ae_all_years_filtered_age.select(f.col('person_id_deid').alias('person_id')).distinct(), 'person_id', how='inner')
)

display(demographics_filtered_age)

# COMMAND ----------

# Filter token pseudo_id table
token_pseudo_id_lookup_filtered_age = (
    token_pseudo_id_lookup_filtered
    .join(hes_ae_all_years_filtered_age.select(f.col('person_id_deid').alias('pseudo_id')).distinct(), 'pseudo_id', how='inner')
)

display(token_pseudo_id_lookup_filtered_age)

# COMMAND ----------

# DBTITLE 1,Add date of death to HES A&E
hes_ae_all_years_filtered_age_with_death = (
    hes_ae_all_years_filtered_age
    .join(demographics_filtered.select(f.col('person_id').alias('person_id_deid'), 'date_of_death', 'in_demo').distinct(), 'person_id_deid', how='left')
    .withColumn('in_demo', f.when(f.col('in_demo').isNotNull(), 1)
                .otherwise(0))
)
display(hes_ae_all_years_filtered_age_with_death)
tab(hes_ae_all_years_filtered_age_with_death, 'in_demo')
tab(hes_ae_all_years_filtered_age_with_death, 'age_at_arrival_date')

# COMMAND ----------

# check
display(hes_ae_all_years_filtered_age_with_death.join(demographics_filtered_age.withColumnRenamed('person_id', 'person_id_deid'), 'person_id_deid', how='left_anti'))

# COMMAND ----------

# DBTITLE 1,Add IMD quintiles 2019 to HES A&E
hes_ae_all_years_filtered_age_with_death_imd = (
    hes_ae_all_years_filtered_age_with_death
    .join(imd_lookup.select('LSOA_2011', 'IMD_2019_QUINTILES').withColumnRenamed('LSOA_2011', 'LSOA11'), 'LSOA11', how='left')
    # .drop('LSOA11', 'lsoa_in_england', 'in_demo', 'PROCODE3', 'PROCODE5')
)

display(hes_ae_all_years_filtered_age_with_death_imd.limit(10))

# COMMAND ----------

# DBTITLE 1,HES APC
# Take latest batch for HES APC
# Include epistart and epiend columns in case Joanna would like to supplement missing admidate fields with these 
hes_apc_all_years_filtered_age = (
    hes_apc_all_years_archive
    .filter(f.col('archived_on') == '2026-01-20')
    .select('person_id_deid',
    'epikey',
    'aekey',
    'admidate',
    'disdate',
    'epistart',
    'epiend',
    'TRETSPEF',
    'FYEAR',
    'LSOA11',
    'ADMIMETH'    
    )
    .filter(f.col('admidate') >= '2012-01-01') # filter to after study start date
 )

display(hes_apc_all_years_filtered_age)

# COMMAND ----------

# Join HES APC to demographics table
hes_apc_all_years_filtered_age_3 = (
    hes_apc_all_years_filtered_age 
    .join(demographics.select(f.col('person_id').alias('person_id_deid'), 'sex_code', 'ethnicity_18_group', 'imd_quintile', 'date_of_birth', 'date_of_death'), ['person_id_deid'], how='left')
    .withColumn('age_at_arrival', f.floor(f.date_diff(f.col('admidate'), f.col('date_of_birth'))/365.25))
    .drop('date_of_birth')
    .filter(f.col('age_at_arrival') >= 18)
)

# COMMAND ----------

tab(hes_apc_all_years_filtered_age_3, 'age_at_arrival')

# COMMAND ----------

# Check min and max admidate
display(hes_apc_all_years_filtered_age_3.select(f.min(f.col('admidate')), f.max(f.col('admidate'))))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ICB

# COMMAND ----------

# Convert ICB lookup to spark dataframe
lsoa_2011_icb = spark.createDataFrame(lsoa_2011_icb)

# COMMAND ----------

# DBTITLE 1,HES AE
# Join hes_ae_all_years_filtered_age_with_death_imd_icb to LSOA-ICB lookup
hes_ae_all_years_filtered_age_with_death_imd_icb = (
    hes_ae_all_years_filtered_age_with_death_imd
    .dropDuplicates()
    .join(lsoa_2011_icb.select(f.col("LSOA11CD").alias('LSOA11'), 'ICB22CD'), 'LSOA11', how='left')
)

# COMMAND ----------

# Check
count_varlist(hes_ae_all_years_filtered_age_with_death_imd_icb, ['person_id_deid', 'aekey', 'arrivaldate', 'ICB22CD'])

# COMMAND ----------

# DBTITLE 1,HES APC
# Join hes apc to ICB
hes_apc_all_years_filtered_age_3_icb = (
    hes_apc_all_years_filtered_age_3
    .dropDuplicates()
    .join(lsoa_2011_icb.select(f.col("LSOA11CD").alias('LSOA11'), 'ICB22CD'), 'LSOA11', how='left')
    .withColumn('ICB22CD_short', f.substring(f.col('ICB22CD'), -2, 2).cast('int')) ## take last two digits of ICB22CD and convert to int
    .withColumn('sex_code', f.col('sex_code').cast('int'))
    .drop('ICB22CD', 'LSOA11')
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create lookup table for ICB22
# MAGIC This will allow for joining of hes_ae and lsoa_multisource tables to the lookup, to identify the ICB22 description (ICB22NM). We have not included them in the original hes_ae and lsoa_multisource tables because researcher is working in Stata and this ICB22NM field is proving difficult to import (memory intensive?)

# COMMAND ----------

icb22_lookup_filtered = (
    lsoa_2011_icb
    .join(hes_ae_all_years_filtered_age_with_death_imd_icb.select(f.col('LSOA11').alias('LSOA11CD')).distinct(), 'LSOA11CD', how='inner')
    .dropDuplicates()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Link HES A&E to ECDS

# COMMAND ----------

# Filter ECDS to the time period where aearrivalmode is mostly 'Unknown' in HES AE 
ecds_filtered = (
    ecds
    .filter(f.col('archived_on') == '2026-01-20')
    .filter((f.col('ARRIVAL_DATE') >= '2020-12-01') & (f.col('ARRIVAL_DATE') <= '2023-12-31'))
    .withColumn('ARRIVAL_MODE_DESC_GROUPED_CODE', f.when(f.col('ARRIVAL_MODE').isin(['1048081000000101', '1048051000000107', '1048041000000109', '1048031000000100', '1048021000000102']), '1')
                .when(f.col('ARRIVAL_MODE').isin(['1048061000000105', '1048071000000103', '1048001000000106',  '1047991000000102']), '2')
                .otherwise('9'))
    .select('PERSON_ID_DEID', f.col('ARRIVAL_DATE').alias('ARRIVALDATE'), f.col('ARRIVAL_TIME').alias('ARRIVALTIME'), 'ARRIVAL_MODE_DESC_GROUPED_CODE')
    # reformat the arrivaltime column to match that in hes_ae
    .withColumn('ARRIVALTIME', f.date_format(f.to_timestamp("ARRIVALTIME", "HH:mm:ss"), "HHmm"))
    .distinct()
)

display(ecds_filtered.limit(10))

## NB: There are records of multiple arrivalmode values for the same arrivaldate and arrivaltime
count_varlist(ecds_filtered, ['PERSON_ID_DEID', 'ARRIVALDATE', 'ARRIVALTIME'])

# COMMAND ----------

display(ecds_filtered.groupBy('PERSON_ID_DEID', 'ARRIVALDATE', 'ARRIVALTIME').agg(f.count("*").alias('count')).where(f.col('count') > 1))

# COMMAND ----------

# Take arrivalmode value at random for such cases
_win1 = Window.partitionBy('PERSON_ID_DEID', 'ARRIVALDATE', 'ARRIVALTIME').orderBy('ARRIVAL_MODE_DESC_GROUPED_CODE')

ecds_filtered = (
    ecds_filtered
    .withColumn('row_num', f.row_number().over(_win1))
    .filter(f.col('row_num') == 1)
)

# check
count_varlist(ecds_filtered, ['PERSON_ID_DEID', 'ARRIVALDATE', 'ARRIVALTIME'])

# COMMAND ----------

# check
tab(ecds_filtered, 'ARRIVAL_MODE_DESC_GROUPED_CODE')

# COMMAND ----------

# Check
count_varlist(hes_ae_all_years_filtered_age_with_death_imd_icb, ['person_id_deid', 'arrivaldate', 'arrivaltime'])

# explore the cases in HES A&E where there are multiple aearrivalmode values (do a display cell below) --> this can inform which to prioritise. 

# COMMAND ----------

# Need to filter hes_ae_all_years_filtered to one row per partition
_win2 = Window.partitionBy('person_id_deid', 'arrivaldate', 'arrivaltime').orderBy(f.col('aearrivalmode').asc_nulls_last())

hes_ae_all_years_filtered_age_with_death_imd_icb = (
    hes_ae_all_years_filtered_age_with_death_imd_icb
    .withColumn('row_num_2', f.row_number().over(_win2))
    .filter(f.col('row_num_2') == 1)
)

# check
tab(hes_ae_all_years_filtered_age_with_death_imd_icb, 'aearrivalmode')

# COMMAND ----------

# 02/03/2026: DECISION TO JOIN HES_AE to ECDS, rather than append. So have HES AE as backbone, and not lose any of the cohort. Appending would be filtering HES AE cohort dataframe to point at which arrivalmode is still relatively complete (lower proportion of 'Unknown' category) and then supplementing with ECDS from that timepoint onwards. Joining is 'stacking' both datasets (performing a left join from HES AE to ECDS)

hes_ae_ecds = merge(hes_ae_all_years_filtered_age_with_death_imd_icb, ecds_filtered, ['person_id_deid', 'arrivaldate', 'arrivaltime'], keep_results=['both', 'left_only'])

display(hes_ae_ecds.limit(10))

# Check
tab(hes_ae_ecds, 'aearrivalmode', 'ARRIVAL_MODE_DESC_GROUPED_CODE')

# COMMAND ----------

# Check
count_varlist(hes_ae_ecds, ['person_id_deid', 'arrivaldate', 'arrivaltime'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Merge hes_ae, demographics, and hes_ae_ecds tables

# COMMAND ----------

# Drop columns not needed from each dataframe
hes_ae_all_years_filtered_age_with_death_imd_icb_final = (
    hes_ae_all_years_filtered_age_with_death_imd_icb
    .drop('LSOA11', 'in_demo', 'PROCODE3', 'PROCODE5', 'date_of_death', 'FYEAR', 'aearrivalmode')
    .dropDuplicates()
)

demographics_filtered_age_final = (
    demographics_filtered_age
    .drop('lsoa', 'imd_quintile', 'in_demo')
    .withColumnRenamed('person_id', 'person_id_deid')
    .dropDuplicates()
)

hes_ae_ecds_final = (
    hes_ae_ecds
    .drop('LSOA11', 'lsoa_in_england', 'in_demo', 'age_at_arrival_date', 'PROCODE3', 'PROCODE5', 'date_of_death', 'ICB22CD', 'IMD_2019_QUINTILES', 'FYEAR', 'row_num', 'has_hes_apc_adm')
    .dropDuplicates()
)

# COMMAND ----------

merged = (
    hes_ae_all_years_filtered_age_with_death_imd_icb_final
    .join(demographics_filtered_age_final, 'person_id_deid', how='left')
    .join(hes_ae_ecds_final, ['person_id_deid', 'aekey', 'arrivaldate', 'arrivaltime'], how='left')
)

# COMMAND ----------

display(merged.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reformat columns

# COMMAND ----------

# Drop columns that are not needed (dropping LSOA, PROCODE3/5, aekey, age_on_arrival, and various flag columns: in_demo, lsoa_in_england)
merged_2 = (
    merged
    .select('person_id_deid', 'date_of_death', 'sex_code', 'ethnicity_18_group', 'age_at_arrival_date', 'arrivaldate', 'arrivaltime', f.col('aearrivalmode').alias('aearrivalmode_hes_ae'), f.col('ARRIVAL_MODE_DESC_GROUPED_CODE').alias('arrival_mode_mapped_ecds'), 'lsoa_in_england', f.col('IMD_2019_QUINTILES').alias('imd_2019_quintiles'), 'ICB22CD', 'id_type', 'has_hes_apc_adm')
)

# COMMAND ----------

display(merged_2.limit(10))

# COMMAND ----------

# Reformat data types of some columns ('destring')
merged_3 = (
    merged_2
    .withColumn('aearrivalmode_hes_ae', f.col('aearrivalmode_hes_ae').cast('int'))
    .withColumn('sex_code', f.col('sex_code').cast('int'))
    .withColumn('arrival_mode_mapped_ecds', f.col("arrival_mode_mapped_ecds").cast('int'))
    .withColumn('ICB22CD_short', f.substring(f.col('ICB22CD'), -2, 2).cast('int')) ## take last two digits of ICB22CD and convert to int
    .drop('ICB22CD')
    .dropDuplicates()
)

display(merged_3.limit(10))

# COMMAND ----------

# Look for duplicates
display(merged_3.groupBy('person_id_deid', 'arrivaldate', 'arrivaltime').count().where(f.col('count') > 1))

# COMMAND ----------

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Check

# COMMAND ----------

# Check
count_var(hes_ae_all_years_filtered_age_with_death_imd_icb, 'person_id_deid')
count_varlist(hes_ae_all_years_filtered_age_with_death_imd_icb, ['person_id_deid', 'arrivaldate'])
count_var(hes_apc_all_years_filtered_age_2, 'person_id_deid')
count_varlist(hes_apc_all_years_filtered_age_2, ['person_id_deid', 'admidate'])
count_var(demographics_filtered_age, 'person_id') 
count_var(token_pseudo_id_lookup_filtered_age, 'pseudo_id')
count_var(hes_ae_ecds, 'person_id_deid')
count_varlist(hes_ae_ecds, ['person_id_deid', 'arrivaldate'])
count_var(merged_3, 'person_id_deid')
count_varlist(merged_3, ['person_id_deid', 'arrivaldate', 'arrivaltime'])

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Save

# COMMAND ----------

save_table(hes_ae_all_years_filtered_age_with_death_imd_icb, f'{proj}_tmp_hes_ae_all_years_cohort', save_previous=False) # also contains IMD and ICB details
save_table(hes_apc_all_years_filtered_age_3_icb, f'{proj}_tmp_hes_apc_all_years', save_previous=True) # this is most recent HES APC output table (not restricted to HES A&E cohort)
save_table(demographics_filtered_age, f'{proj}_tmp_demographics', save_previous=False)
save_table(token_pseudo_id_lookup_filtered_age, f'{proj}_tmp_token_pseudo_id_lookup', save_previous=False)
save_table(hes_ae_ecds, f'{proj}_tmp_hes_ae_all_years_cohort_with_ecds', save_previous=False)

# Save merged table 
save_table(merged_3, f'{proj}_tmp_merged', save_previous=False)

save_table(icb22_lookup_filtered, f'{proj}_icb_lookup', save_previous=False)
