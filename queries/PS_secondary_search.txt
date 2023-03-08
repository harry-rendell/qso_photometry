-- This query takes objID_ps from PS_neighbour_search and finds all observations associated with those IDs.
select
    q.uid, d.objID, d.detectID, filter=f.filterType, d.obsTime, d.ra, d.dec,
    d.psfFlux, d.psfFluxErr
into mydb.ps_secondary
from mydb.dr14q_ps_neighbours q
join Detection d on d.objid = q.objid_ps
join Filter f on f.filterID = d.filterID
order by uid ASC, obsTime ASC