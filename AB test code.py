#%%
import pandas as pd
import numpy as np
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
#%%
df = pd.read_csv('ab_test_click_data.csv')
#%%
df.head()
#%%
df.describe()
#%%
df.groupby("group")["click"].sum()
#%%
palette={0:"yellow",1:"black"}
plt.figure(figsize=(10,6))
ax=sns.countplot(x="group", data=df, hue="click", palette=palette)
plt.title("Click Distribution in Experimental and Control groups")
plt.xlabel("Group")
plt.ylabel("Cont")
plt.legend(title="Click", labels=["No", "Yes"])
#%%
# Asignar paleta de colores
palette = {0: 'orange', 1: 'black'}

# Graficar
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='group', hue='click', data=df, palette=palette)
plt.title("Distribución de Clicks en los grupos de control y experimental")
plt.xlabel("Grupo")
plt.ylabel("Conteo de Clicks")
plt.legend(title="Click", labels=["No", "Sí"])  # Cambiar el argumento 'label' a 'labels'

# Calcular los porcentajes
group_counts = df.groupby(["group"]).size()
group_click_counts = df.groupby(["group", "click"]).size().reset_index(name="count")
for p in ax.patches:                # Grafikteki her bir sütun (çubuk) için döngüyü çalıştır
    height = p.get_height()          # Sütunun yüksekliğini (tıklama sayısını) al

    # --- Kritik Mantık: Hangi çubuk hangi gruba ait? ---
    # p.get_x() sütunun sol kenar koordinatını verir.
    # Bu grafik muhtemelen x=0 ve x=1 civarında iki gruba ayrılmış.

    # Çubuğun x konumu 0.5'ten küçükse deney grubudur, değilse kontroldür.
    group = "exp" if p.get_x() < 0.5 else "con"

    # Çubuğun x konumu 0.5'ten büyükse tıklama (1), değilse tıklamama (0) sütunudur.
    click = 1 if p.get_x() > 0.5 else 0

    # Sütunun ait olduğu grubun toplam eleman sayısını al.
    total = group_counts[group]

    # Yüzde hesabı: (Sütun Yüksekliği / Grup Toplamı) * 100
    percentage = height / total * 100

    # --- Metni Grafiğe Ekleme ---
    ax.text(
        # X Konumu: Sütunun tam ortası (Sol kenar + Genişliğin yarısı)
        p.get_x() + p.get_width() / 2,

        # Y Konumu: Sütunun tepesinin 5 birim üstü (üst üste binmesin diye)
        height + 5,

        # İçerik: Hesaplanan yüzde, 1 ondalık basamakla (örneğin: "50.7%")
        f'{percentage:.1f}%',

        # Hizalama: Yatayda merkezle
        ha="center",

        color="black", # Yazı rengi
        fontsize=10,   # Yazı boyutu
    )
#%%
N_con = df[df["group"] == "con"].count()
N_exp = df[df["group"] == "exp"].count()

# calculating the total number of clicks per group by summing 1's
X_con = df.groupby("group")["click"].sum().loc["con"]
X_exp = df.groupby("group")["click"].sum().loc["exp"]

# printing this for visibility
print(df.groupby("group")["click"].sum())
print("Number of user in Control: ", N_con)
print("Number of users in Experimental: ", N_exp)
print("Number of CLicks in Control: ", X_con)
print("Number of CLicks in Experimental: ", X_exp)
#%%
# Her grup için tıklama olasılığı tahminini hesaplama
p_con_hat = X_con / N_con
p_exp_hat = X_exp / N_exp

print("Click Probability in Control Group:", p_con_hat)
print("Click Probability in Experimental Group:", p_exp_hat)

# Havuzlanmış (pooled) tıklama olasılığı tahminini hesaplama
p_pooled_hat = (X_con + X_exp) / (N_con + N_exp)
print("Pooled Click Probability:", p_pooled_hat)
#%%
# --- ADIM 3: Varyansın Hesaplanması (Resim 2'deki Kod) ---
# İstatistiksel anlamlılık testi (z-testi) yapabilmek için
# havuzlanmış varyansı hesaplamamız gerekiyor.

# Havuzlanmış varyans formülü uygulanıyor
pooled_variance = p_pooled_hat * (1 - p_pooled_hat) * (1/N_con + 1/N_exp)

print("Havuzlanmış Varyans: ", pooled_variance)
#%%
alpha=0.05
delta=0.1
# computing the standard error of the test
# (testin standart hatasının hesaplanması)
SE = np.sqrt(pooled_variance)
print("Standard Error is: ", SE)

# computing the test statistics of Z-test
# (Z-testi istatistiğinin hesaplanması)
Test_stat = (p_con_hat - p_exp_hat)/SE
print("Test Statistics for 2-sample Z-test is:", Test_stat)

# critical value of the Z-test
# (Z-testinin kritik değerinin belirlenmesi)
Z_crit = norm.ppf(1-alpha/2)
print("Z-critical value from Standard Normal distribution: ", Z_crit)
#%%
# calculating p value
# (p değerinin hesaplanması)
p_value = 2 * norm.sf(abs(Test_stat))

# function checking the statistical significance
# (istatistiksel anlamlılığı kontrol eden fonksiyon)
def is_statistical_significance(p_value, alpha):
    """
    We assess whether there is statistical significance based on the p-value and alpha.
    (P-değeri ve alpha'ya dayanarak istatistiksel anlamlılık olup olmadığını değerlendiririz.)
    Arguments:
    - p_value (float): The p-value resulting from a statistical test.
    - alpha (float, optional): The significance level threshold used to determine statistical significance.
    Returns:
    - Prints the assessment of statistical significance.
    """
    # Print the rounded p-value to 3 decimal places
    # (P-değerini virgülden sonra 3 basamağa yuvarlayarak yazdır)
    print(f"P-value of the 2-sample Z-test: {round(p_value, 3)}")

    # Determine statistical significance
    # (İstatistiksel anlamlılığı belirle)
    if p_value < alpha:
        print("There is statistical significance, indicating that the observed differences between the groups are unlikely to have occurred by chance.")
    else:
        print("There is no statistical significance, suggesting that the observed differences between the groups could have occurred by chance.")

#%%

#%%
