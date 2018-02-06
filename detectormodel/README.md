# Example geometry

COSILike.geo.setup is a simplified, COSI-like geometry for all simulations 

You can take a look at the geometry via:
```
geomega -f COSILike.geo.setup
```
Click the play button and the geometry should pop up.

To test the geometry, first run a simple simulation of a 511-keV point source:
```
cosima COSILike.source
```

Then reconstruct the Compton data
```
revan -a -f COSILike.inc1.id1.sim -g COSILike.geo.setup -c COSILike.revan.cfg
```

And finally look at the image:
```
mimrec -f COSILike.inc1.id1.tra -g COSILike.geo.setup -c COSILike.mimrec.cfg
```
Click the play button and the image should pop up.

