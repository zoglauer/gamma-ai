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

    geometries = [si_geo, btm_cal_geo, low_negX_cal_geo, low_negY_cal_geo, low_posX_cal_geo, low_posY_cal_geo,
                  high_negX_cal_geo, high_negY_cal_geo, high_posX_cal_geo, high_posY_cal_geo]

    CsI_radiation_length = 1.86 # [cm] @source https://pdg.lbl.gov/2022/AtomicNuclearProperties/HTML/cesium_iodide_CsI.html

    Si_radiation_length = 9.37 # [cm] @source https://pdg.lbl.gov/2022/AtomicNuclearProperties/HTML/silicon_Si.html

    cal_x0 = (1 / 0.9) * CsI_radiation_length

    tracker_x0 = Si_radiation_length * 10

    energy_crit_Si = 40.19 # [MeV] @source https://pdg.lbl.gov/2022/AtomicNuclearProperties/HTML/silicon_Si.html

    energy_crit_CsI = 11.17 # [MeV] @source https://pdg.lbl.gov/2022/AtomicNuclearProperties/HTML/cesium_iodide_CsI.html

    effective_E_crit_tracker = 0.4 # [MeV] Ec2 = (Zeff / ZSi)2 x Ec1 @source ChatGPT

    effective_E_crit_cal = 9.05 # [MeV] Ec2 = (Zeff / ZCs)^2 x Ec1 @source ChatGPT

    @staticmethod
    def verifyHit(hit):
        """
        A function that checks whether the given hit was in the bounds of a calorimeter.
        @param hit : x = hit[0], y = hit[1], z = hit[2]
        @return 1 if in bounds, 0 if out of bounds
        """
        return DetectorGeometry.cordsInGeo(hit, range(len(DetectorGeometry.geometries))) and 1 or 0


    @staticmethod
    def cordsInGeo(cords, indexRange):

        x, y, z = cords[0], cords[1], cords[2]

        for index in indexRange:
            x_range, y_range, z_range = DetectorGeometry.geometries[index]
            if x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1] and z_range[0] <= z <= z_range[1]:
                return True

        return False

    @staticmethod
    def radLength(x, y, z):

        # if in calorimeter(s)
        if DetectorGeometry.cordsInGeo([x, y, z], range(1, len(DetectorGeometry.geometries))):
            return DetectorGeometry.cal_x0

        # must be in tracker if not in calorimeter(s)
        return DetectorGeometry.tracker_x0

    @staticmethod
    def critE(x, y, z):

        # if in tracker
        if DetectorGeometry.cordsInGeo([x, y, z], 0):
            return DetectorGeometry.effective_E_crit_tracker

        # must be in calorimeter(s) if not in tracker
        return DetectorGeometry.effective_E_crit_cal
