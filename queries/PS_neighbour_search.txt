-- This query searches for Pan-STARRS object IDs which correspond to the DR14 quasars, and saves them as objID_ps
-- Note that we use fGetNearbyObjEq, which may return multiple objID_ps for a given uid. This is because the PS stacking is not perfect.
select

q.uid as uid, o.objID as objID_ps, 
q.ra, q.dec,
nb.distance as sep

into mydb.dr14q_ps_neighbours
from mydb.dr14q_uid_coords q

cross apply dbo.fGetNearbyObjEq(q.ra, q.dec, 1.0/60) as nb
join StackObjectThin o on o.objid = nb.objid where o.primaryDetection = 1

order by uid ASC