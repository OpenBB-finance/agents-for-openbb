
CREATE OR REPLACE CORTEX SEARCH SERVICE cortex_search_tutorial_db.public.fomc_meeting
    ON chunk
    ATTRIBUTES language
    WAREHOUSE = cortex_search_tutorial_wh
    TARGET_LAG = '1 hour'
    AS (
    SELECT
        chunk,
        relative_path,
        file_url,
        language
    FROM cortex_search_tutorial_db.public.docs_chunks_table
    );