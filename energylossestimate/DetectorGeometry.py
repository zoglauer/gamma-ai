class DetectorGeometry:
    """
    A class that stores the geometry and radiation lengths of the tracker and calorimeter(s). 
    @geo [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
    units - centimeters
    """

    si_geo = ((-45.8, 45.8), (-48.3, 48.3), (10.2, 45))

    btm_cal_geo = ((-50.5, 50.5), (-50.5, 50.5), (2.13, 10.13))

    low_negX_cal_geo = ((-46, -50), (-48.175, 48.175), (10.58, 28.74))
    low_posX_cal_geo = ((46, 50), (-48.175, 48.175), (10.58, 28.74))

    low_negY_cal_geo = ((-48.175, 48.175), (48.5, 52.5), (10.58, 28.74))
    low_posY_cal_geo = ((-48.175, 48.175), (-48.5, -52.5), (10.58, 28.74))

    high_negX_cal_geo = ((-46, -48), (-48.175, 48.175), (10.58, 28.74))
    high_posX_cal_geo = ((46, 58), (-48.175, 48.175), (10.58, 28.74))

    high_negY_cal_geo = ((-48.175, 48.175), (48.5, 50.5), (10.58, 28.74))
    high_posY_cal_geo = ((-48.175, 48.175), (-48.5, -50.5), (10.58, 28.74))

    CsI_x0 = 1.85

    Si_x0 = 0.937

    cal_x0 = (1 / 0.9) * CsI_x0

    tracker_x0 = 10 * Si_x0

    @staticmethod
    def verifyHit(hit):
        """
        A function that checks whether the given hit was in the bounds of a calorimeter.
        @param hit : x = hit[0], y = hit[1], z = hit[2]
        @return 1 if in bounds, 0 if out of bounds
        """

        def cordsInGeo(cords, geo):
            x, y, z = cords[0], cords[1], cords[2]
            x_range, y_range, z_range = geo

            if x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1] and z_range[0] <= z <= z_range[1]:
                return True

            return False

        return any((cordsInGeo(hit, DetectorGeometry.low_negX_cal_geo),
                    cordsInGeo(hit, DetectorGeometry.low_posX_cal_geo),
                    cordsInGeo(hit, DetectorGeometry.low_negY_cal_geo),
                    cordsInGeo(hit, DetectorGeometry.low_posY_cal_geo),
                    cordsInGeo(hit, DetectorGeometry.high_negX_cal_geo),
                    cordsInGeo(hit, DetectorGeometry.high_posX_cal_geo),
                    cordsInGeo(hit, DetectorGeometry.high_negY_cal_geo),
                    cordsInGeo(hit, DetectorGeometry.high_posY_cal_geo),
                    cordsInGeo(hit, DetectorGeometry.btm_cal_geo),
                    cordsInGeo(hit, DetectorGeometry.si_geo)
                    )) and 1 or 0

    @staticmethod
    def radLengthForZ(z):
        return DetectorGeometry.tracker_x0 if z - 10.58 > 0 else DetectorGeometry.cal_x0