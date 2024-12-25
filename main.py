import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Veri setini yükleme için sütun isimleri
columns = [
    "state", "county", "community", "communityname", "fold", "population", "householdsize", "racepctblack",
    "racePctWhite", "racePctAsian", "racePctHisp", "agePct12t21", "agePct12t29", "agePct16t24", "agePct65up",
    "numbUrban", "pctUrban", "medIncome", "pctWWage", "pctWFarmSelf", "pctWInvInc", "pctWSocSec", "pctWPubAsst",
    "pctWRetire", "medFamInc", "perCapInc", "whitePerCap", "blackPerCap", "indianPerCap", "AsianPerCap",
    "OtherPerCap", "HispPerCap", "NumUnderPov", "PctPopUnderPov", "PctLess9thGrade", "PctNotHSGrad", "PctBSorMore",
    "PctUnemployed", "PctEmploy", "PctEmplManu", "PctEmplProfServ", "PctOccupManu", "PctOccupMgmtProf",
    "MalePctDivorce", "MalePctNevMarr", "FemalePctDiv", "TotalPctDiv", "PersPerFam", "PctFam2Par", "PctKids2Par",
    "PctYoungKids2Par", "PctTeen2Par", "PctWorkMomYoungKids", "PctWorkMom", "NumIlleg", "PctIlleg", "NumImmig",
    "PctImmigRecent", "PctImmigRec5", "PctImmigRec8", "PctImmigRec10", "PctRecentImmig", "PctRecImmig5",
    "PctRecImmig8", "PctRecImmig10", "PctSpeakEnglOnly", "PctNotSpeakEnglWell", "PctLargHouseFam",
    "PctLargHouseOccup", "PersPerOccupHous", "PersPerOwnOccHous", "PersPerRentOccHous", "PctPersOwnOccup",
    "PctPersDenseHous", "PctHousLess3BR", "MedNumBR", "HousVacant", "PctHousOccup", "PctHousOwnOcc",
    "PctVacantBoarded", "PctVacMore6Mos", "MedYrHousBuilt", "PctHousNoPhone", "PctWOFullPlumb", "OwnOccLowQuart",
    "OwnOccMedVal", "OwnOccHiQuart", "RentLowQ", "RentMedian", "RentHighQ", "MedRent", "MedRentPctHousInc",
    "MedOwnCostPctInc", "MedOwnCostPctIncNoMtg", "NumInShelters", "NumStreet", "PctForeignBorn", "PctBornSameState",
    "PctSameHouse85", "PctSameCity85", "PctSameState85", "LemasSwornFT", "LemasSwFTPerPop", "LemasSwFTFieldOps",
    "LemasSwFTFieldPerPop", "LemasTotalReq", "LemasTotReqPerPop", "PolicReqPerOffic", "PolicPerPop",
    "RacialMatchCommPol", "PctPolicWhite", "PctPolicBlack", "PctPolicHisp", "PctPolicAsian", "PctPolicMinor",
    "OfficAssgnDrugUnits", "NumKindsDrugsSeiz", "PolicAveOTWorked", "LandArea", "PopDens", "PctUsePubTrans",
    "PolicCars", "PolicOperBudg", "LemasPctPolicOnPatr", "LemasGangUnitDeploy", "LemasPctOfficDrugUn",
    "PolicBudgPerPop", "ViolentCrimesPerPop"
]

# Veri setini yükle
file_path = "communities.data"
data = pd.read_csv(file_path, header=None, names=columns, na_values="?")

# Eksik verileri çıkar
# Eksik değerler varsa çıkarıyoruz.
data = data.dropna()

# Hedef değişkeni oluştur (Binary Classification)
# Şiddet suç oranı için eşik değer kullanarak sınıflandırma yapıyoruz.
threshold = 0.1
data['target'] = (data['ViolentCrimesPerPop'] > threshold).astype(int)

# Özellik ve hedef değişkenlerini ayırıyoruz.
X = data.drop(columns=['state', 'county', 'community', 'communityname', 'fold', 'ViolentCrimesPerPop', 'target'])
y = data['target']

# Eğitim ve test verilerini ayırma
# Eğitim ve test verilerinin %70/%30 oranında bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model oluştur ve eğit (Sınıflandırma)
# RandomForestClassifier modelini kullanarak eğitiyoruz.
model_classification = RandomForestClassifier(random_state=42)
model_classification.fit(X_train, y_train)

# Test verisi üzerinde tahmin yapma
# Test verisi ile sınıflandırma tahminleri alıyoruz.
y_pred = model_classification.predict(X_test)

# Performans metriklerini hesaplama (Sınıflandırma)
# Doğru tahmin oranını hesaplar
accuracy = accuracy_score(y_test, y_pred)
# Pozitif tahminlerin doğruluk oranını hesaplar
precision = precision_score(y_test, y_pred)
# Gerçek pozitiflerin tespit oranını hesaplar
recall = recall_score(y_test, y_pred)
# Precision ve Recall'un harmonik ortalamasını hesaplar
f1 = f1_score(y_test, y_pred)
# Confusion matrix ile sınıflandırma hatalarını analiz eder
conf_matrix = confusion_matrix(y_test, y_pred)

# Pozitif sınıfın doğru tespit edilme oranı (duyarlılık)
sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
# Negatif sınıfın doğru tespit edilme oranı (özgüllük)
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])

# Performans metriklerini ekrana yazdırma
print("--- Sınıflandırma Performansı ---")
print(f"Accuracy: {accuracy:.2f}")  # Modelin genel doğruluğu
print(f"Precision: {precision:.2f}")  # Pozitif tahminlerdeki doğruluk
print(f"Recall: {recall:.2f}")  # Pozitif sınıfların yakalanma oranı
print(f"F1 Score: {f1:.2f}")  # Precision ve Recall'un dengesi
print(f"Sensitivity: {sensitivity:.2f}")  # Pozitif sınıf tespiti başarısı
print(f"Specificity: {specificity:.2f}")  # Negatif sınıf tespiti başarısı
