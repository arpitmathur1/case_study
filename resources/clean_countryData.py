import pyodbc
import pandas as pd


conn = pyodbc.connect(r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=H:\project\EY\EY_case_study\data_external\JPHarvestField.accdb;')
cursor = conn.cursor()
cursor.execute("""select * from tblGEO3Countries""")

data = []
for row in cursor.fetchall():
    data.append(list(row))


colName = ["ROG3","ISO3","ISO2","OWCountryCode","AD2000CountryCode","EthnologueCountryCode","CountryNumber","Ctry","CtryShort","AltName","CountryNameSpanish","CountryNameSpanishShort","CountryNameGerman","CountryNameGermanShort","CountryNamePortuguese","CountryNamePortugueseShort","CountryNameFrench","CountryNameFrenchShort","CountryNameIndonesian","CountryNameIndonesianShort","CountryNameChinese","CountryNameChineseShort","CountryNameKorean","CountryNameKoreanShort","CountryNameRussian","CountryNameRussianShort","CountryNameHindi","CountryNameVietnamese","CountryNameVietnameseShort","CountryNameDutch","GermanConcatenationPrefix","FrenchConcatenationPrefix","PortugueseConcatenationPrefix","SpanishConcatenationPrefix","IndonesianConcatenationPrefix","ChineseConcatenationPrefix","KoreanConcatenationPrefix","RussianConcatenationPrefix","VietnameseConcatenationPrefix","EthnologueMapExists","LibraryCongressReportExists","StateDeptReligiousFreedom","SouthAsia","ROG2","RegionCodeJPOld","RegionCode","UNSR","WCDSubRegionCode","MANIRegionCode","Capital","Population","PopulationSource","PoplGrowthRate","PercentUrbanized","PercentUnderAge15","V59Country","OWLaunchFocus","PrayercastVideoActive","PrayercastVideo","PrayercastVimeoURL","GodReportsURL","BibleSocietyURL","CorruptionLevel","EthnolinguisticMap","UNMap","AreaSquareMiles","AreaSquareKilometers","PopulationPerSquareMile","CountryPhoneCode","SecurityLevel","PersecutionRankingOD","PersecutionRankingODLink","IsCountry","JPScale","JPScaleProgressMap","10/40Window","10/40WindowOriginal","PostChristian","WINCountryProfile","BJMFocusCountry","OWPctChristian1986","OWPctEvangelical1986","OWPctChristian1993","OWPctEvangelical1993","OWPctChristian2001","OWPctEvangelical2001","OWPctEvang","OWPctCharismatic","OWEvangelicalGrowthRate","StonyGround","USAPostalSystem","LiteracyRate","LiteracySource","ROL3OfficialLanguage","ROL3SecondaryLanguage","RLG3Primary","RLG4Primary","PctAnimist","PctBahai","PctBuddhism","PctChinese","PctHindu","PctIslam","PctSikh","PctJudaism","PctNonReligious","PctOther","PctChristian","PctChristianProtestant","PctChristianIndependent","PctChristianAnglican","PctChristianRomanCatholic","PctChristianOrthodoxCatholic","PctChristianMarginal","PctChristianUnaffiliated","PctChristianDoublyProfessing","PctChristianOther","PctChristianExtra","PctProtestantJP","PctIndependentJP","PctAnglicanJP","PctRomanCatholicJP","PctOrthodoxJP","PctOtherJP","ReligionDataYear","OWDate","ROP1","NSMMissionArticles","HDIYear","HDIValue","HDIRank","InternetCtryCode","InternetUsers%","FlashMapZoom","FlashMapZoomX","FlashMapZoomY","GoogleMapZoom","GoogleMapLat","GoogleMapLng","Source","EditName","EditDate"]
finalColNamesNeeded = ["ISO3","ISO2","Ctry","AreaSquareMiles","PopulationPerSquareMile","PoplGrowthRate","PercentUrbanized","PercentUnderAge15","Population","LiteracyRate","LiteracySource","ROL3OfficialLanguage","ROL3SecondaryLanguage","RLG3Primary","RLG4Primary","PctAnimist","PctBahai","PctBuddhism","PctChinese","PctHindu","PctIslam","PctSikh","PctJudaism","PctNonReligious","PctOther","PctChristian","PctChristianProtestant","PctChristianIndependent","PctChristianAnglican","PctChristianRomanCatholic","PctChristianOrthodoxCatholic","PctChristianMarginal","PctChristianUnaffiliated","PctChristianDoublyProfessing","PctChristianOther","PctChristianExtra","PctProtestantJP","PctIndependentJP","PctAnglicanJP","PctRomanCatholicJP","PctOrthodoxJP","PctOtherJP","ReligionDataYear"]
df = pd.DataFrame(data, columns=colName)

cols = list(set(list(df.columns)) - set(finalColNamesNeeded))
cols.append(['PctProtestantJP', 'PctIndependentJP','PctAnglicanJP', 'PctRomanCatholicJP', 'PctOrthodoxJP', 'PctOtherJP','ReligionDataYear'])
for col in cols:
    df.drop(col, axis=1, inplace=True)

df.set_index('Ctry', inplace=True)
df.drop(["Jarvis Island","Antarctica","Navassa Island","Baker Island","Howland Island","Johnson Atoll","Kingman Reef","Palmyra Atoll","Coral Sea Islands","Ashmore and Cartier Islands","Bouvet Island","Europa Island","Guernsey","Glorioso Islands","Gaza Strip","Heard and McDonald Islands","Clipperton Island","Jersey","Jan Mayen","Juan de Nova Island","Midway Island","No Man's Land","Netherlands Antilles","Paracel Islands","Spratly Islands","Serbia (archived)","South Georgia and the South Sandwich Islands","Tromelin Island","Wake Island","Serbia and Montenegro (archived)"], axis=0, inplace=True)

print(df.head())
print(df.describe())
df.to_csv('../data_external/countryData.csv')
