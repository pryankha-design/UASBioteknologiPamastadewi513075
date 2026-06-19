import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==========================================
# 1. DEFINISI PARAMETER (Sesuai Tabel Soal)
# ==========================================
V1_max = 5.0
Km1 = 2.0
Ki = 3.0
X = 10.0  # Konsentrasi substrat eksternal konstan
k2 = 1.0  # Konstanta laju orde satu untuk A -> B
k3 = 0.8  # Konstanta laju orde satu untuk B -> P
k4 = 0.3  # Konstanta laju orde satu untuk A -> Byproduct

# ==========================================
# 2. DEFINISI SISTEM PERSAMAAN DIFERENSIAL (ODEs)
# ==========================================
def kinetic_system(t, y):
    # y[0] = [A], y[1] = [B], y[2] = [P], y[3] = [Byproduct]
    A, B, P, Byproduct = y
    
    # Persamaan laju reaksi v1 dengan Non-Competitive Inhibition dari P
    v1 = (V1_max * X) / ((Km1 + X) * (1 + (P / Ki)))
    
    # Persamaan laju reaksi orde satu untuk jalur berikutnya
    v2 = k2 * A
    v3 = k3 * B
    v4 = k4 * A
    
    # Mass balance / ODEs (berdasarkan matriks stoikiometri sebelumnya)
    dAdt = v1 - v2 - v4
    dBdt = v2 - v3
    dPdt = v3
    dByproductdt = v4
    
    return [dAdt, dBdt, dPdt, dByproductdt]

# ==========================================
# 3. KONDISI AWAL & RENTANG WAKTU SIMULASI
# ==========================================
# Asumsi awal semua konsentrasi metabolit internal di dalam sel adalah 0
initial_conditions = [0.0, 0.0, 0.0, 0.0]  # [A_0, B_0, P_0, Byproduct_0]

# Rentang waktu simulasi (t = 0 hingga t = 20)
t_span = (0, 20)
t_eval = np.linspace(t_span[0], t_span[1], 500)  # Menyediakan 500 titik data agar grafik halus

# ==========================================
# 4. MENJALANKAN SIMULASI (NUMERICAL INTEGRATION)
# ==========================================
sol = solve_ivp(kinetic_system, t_span, initial_conditions, t_eval=t_eval, method='RK45')

# ==========================================
# 5. VISUALISASI HASIL (PLOTTING GRAPH)
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label='Metabolite A', linewidth=2)
plt.plot(sol.t, sol.y[1], label='Metabolite B', linewidth=2)
plt.plot(sol.t, sol.y[2], label='Product P', linewidth=2, color='red', linestyle='--')
plt.plot(sol.t, sol.y[3], label='Byproduct', linewidth=2, linestyle=':')

plt.title('Simulasi Dinamika Kinetik Jalur Metabolisme (Allosteric Inhibition)', fontsize=14, fontweight='bold')
plt.xlabel('Waktu (t)', fontsize=12)
plt.ylabel('Konsentrasi', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()

# Menampilkan grafik di layar
plt.show()
