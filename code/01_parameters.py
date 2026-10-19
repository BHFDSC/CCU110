# Databricks notebook source
import pyspark.sql.functions as f
import pyspark.sql.types as t
import pandas as pd
import re
import datetime

# COMMAND ----------

# -----------------------------------------------------------------------------
# Databases
# -----------------------------------------------------------------------------
db = ''
dbc = f''
dsa = f''
dss = f''

# COMMAND ----------

# Project specific parameter
proj = 'ccu110_01'

# COMMAND ----------

