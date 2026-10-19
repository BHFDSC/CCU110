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

hes_apc_cohort = spark.table(f'{dsa}.ccu110_01_tmp_hes_apc_all_years')
cips = spark.table(f'{dsa}.hds_curated_assets__hes_apc_cips_episodes_2026_01_20')

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Prepare

# COMMAND ----------

cips_filtered = (
    cips
    .select(f.col('person_id').alias('person_id_deid'), 'epikey', 'cips_spell_count', 'cips_epistart', 'cips_epiend','cips_admidate', 'cips_disdate')
)

# COMMAND ----------

hes_apc_cohort_with_cips = (
    hes_apc_cohort
    .join(cips_filtered, ['person_id_deid','epikey'], how = 'left')
)

# COMMAND ----------

display(hes_apc_cohort_with_cips.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Check

# COMMAND ----------

count_varlist(hes_apc_cohort_with_cips, ['person_id_deid', 'disdate'])
count_varlist(hes_apc_cohort_with_cips, ['person_id_deid', 'cips_disdate'])

# COMMAND ----------

# Flag where cips and hes_apc disdate differs
tmp1 = (
    hes_apc_cohort_with_cips
    .withColumn('disdate_diff', f.when(f.col('disdate') == f.col('cips_disdate'), f.lit(0))
                .when(f.col('disdate') != f.col('cips_disdate'), f.lit(1))
                .otherwise(None))
    )

tab(tmp1, 'disdate_diff')

# COMMAND ----------

# check if any records where disdate is not null and cips disdate is null
display(tmp1.filter(f.col('disdate').isNotNull() & f.col('cips_disdate').isNull()).select(f.min(f.col('disdate')), f.max(f.col('disdate'))))

# Number of records where cips disdate is null
print(f"Number of records where cips disdate is null: {tmp1.filter(f.col('cips_disdate').isNull()).count()}")

# Number of records where disdate is null
print(f"Number of records where HES APC disdate is null: {tmp1.filter(f.col('disdate').isNull()).count()}")

# Number of records where both null
print(f"Number of records where BOTH disdate is null: {tmp1.filter(f.col('disdate').isNull() & f.col('cips_disdate').isNull()).count()}")

# Number of records where cips is null but not hes apc disdate
print(f"Number of records where HES APC disdate is not null but CIPS disdate is null: {tmp1.filter(f.col('disdate').isNotNull() & f.col('cips_disdate').isNull()).count()}")

# Number of records where cips is not null but hes apc disdate is
print(f"Number of records where HES APC disdate is null but CIPS disdate is not null: {tmp1.filter(f.col('disdate').isNull() & f.col('cips_disdate').isNotNull()).count()}")


# COMMAND ----------

tab(tmp1.filter(f.col('disdate').isNotNull() & f.col('cips_disdate').isNull()), 'disdate')

# COMMAND ----------

tab(tmp1.filter(f.col('disdate_diff') == 1), 'disdate')

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Save

# COMMAND ----------

save_table(hes_apc_cohort_with_cips, out_name= f'{proj}_tmp_hes_apc_all_years_with_cips', save_previous=True)

# COMMAND ----------

