-- This query takes the DR14Q coords and finds the nearest associated objects within 3"
-- The output is not used to find secondary observations, it is just used as a check to see the kinds of objects returned.
select

q.uid as uid, p.objID, 
p.ra as ra, p.dec as dec,
q.ra as ra_ref, q.dec as dec_ref,
s.class as class,
nb.distance as get_nearby_distance

into mydb.dr14q_nearby_w_class
from mydb.dr14q_uid_coords q
cross apply dbo.fGetNearbyObjEq(q.ra, q.dec, 0.05) as nb

join PhotoPrimary p on p.objid=nb.objid
join specobj as s on s.bestobjid=p.objid