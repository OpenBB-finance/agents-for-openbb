CREATE DATABASE IF NOT EXISTS cortex_search_tutorial_db;

CREATE OR REPLACE WAREHOUSE cortex_search_tutorial_wh WITH
     WAREHOUSE_SIZE='X-SMALL'
     AUTO_SUSPEND = 120
     AUTO_RESUME = TRUE
     INITIALLY_SUSPENDED=TRUE;

 USE WAREHOUSE cortex_search_tutorial_wh;