-- This query takes the DR14Q coordinates and finds all secondary observations within 1 arcsecond
-- We name the output sdss_secondary_unmelted which is processed by /pipeline/parsing_SDSS.ipynb

select

q.uid as uid, p.objID,
q.ra, q.dec,
nb.distance as get_nearby_distance,

p.psfMag_u as mag_u, p.psfMagErr_u as magerr_u,
p.psfMag_g as mag_g, p.psfMagErr_g as magerr_g,
p.psfMag_r as mag_r, p.psfMagErr_r as magerr_r,
p.psfMag_i as mag_i, p.psfMagErr_i as magerr_i,
p.psfMag_z as mag_z, p.psfMagErr_z as magerr_z,
f.mjd_r

into mydb.dr14q_secondary
from mydb.dr14q_uid_coords q
cross apply dbo.fGetNearbyObjAllEq(q.ra, q.dec, 1.0/60) as nb

join photoobj p on p.objid=nb.objid
join field f on f.fieldid=p.fieldid
  
ORDER BY uid ASC, mjd_r ASC