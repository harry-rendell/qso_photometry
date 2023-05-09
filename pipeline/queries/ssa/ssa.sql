-- 1. First get plate_table.csv by running this query on ssa.roe.ac.uk/sql.html.
--      Remove first line of output as it is null.

select
    plateID,
    surveyID,
    fieldID,
    filterID,
    utDateObs

from Plate

ORDER BY surveyID, plateID

-- 2. Upload chunks of coordinates to ssa.roe.ac.uk/xmatch.html
--      Pairing radius = 1" (note: entry field is in arcmin)
--      Paring option = All nearby (if we choose Nearest then we only get 1 filterband per object)
--      Data Format = ASCII
--      Select: objID,ssaField,surveyID,plateID,smag
--      From: Detection (use Detection instead of source so that we have a breakdown of filterbands)
