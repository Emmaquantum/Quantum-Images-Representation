# Importaciones de librerías comunes
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

# Importaciones de Qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram

# ## CORRECCIÓN ##: Eliminado 'Session', que no se usa en el plan gratuito.
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit_ibm_runtime.exceptions import IBMRuntimeError


class FRQI_MARY_Simulator:
    """
    Simulador del algoritmo FRQI (Flexible Representation of Quantum Images)
    utilizando una implementación personalizada de la puerta multicontrolada.
    """

    def __init__(self, intensities=None, max_intensity=255, shots=8000, n=2):
        self.service = None
        self.backend = None
        self.n = n
        self.max_intensity = max_intensity
        self.shots = shots
        self.intensities = intensities if intensities is not None else [15, 25, 150, 255]
        self.thetas = self.calculate_thetas(self.intensities)
        self.results = {}
        self.counts = None
        self.qc = None

    def calculate_thetas(self, intensities):
        """
        ## CORRECCIÓN ##: Se ajustó la fórmula de normalización y de cálculo de thetas
        a la implementación estándar de FRQI.
        """
        # Normaliza intensidades al rango [0, 1]
        intensity_norm = [min(1.0, max(0.0, i / self.max_intensity)) for i in intensities]
        # Calcula thetas con la fórmula FRQI: theta = 2 * arcsin(sqrt(intensidad_normalizada))
        thetas = [2 * np.arcsin(np.sqrt(i)) for i in intensity_norm]
        return thetas

    def connect_to_ibm_backend(self, token: str, backend_name: str = None, channel: str = "ibm_quantum_platform"):
        """
        Conecta al servicio IBM Quantum y guarda el objeto backend.
        """
        try:
            self.service = QiskitRuntimeService(channel=channel, token=token)
            print(f"✅ Conexión establecida con IBM Quantum Platform (canal: {channel}).")

            if backend_name:
                self.backend = self.service.backend(backend_name)
            else:
                backends = self.service.backends(simulator=False, operational=True)
                real_backends = [b for b in backends if b.num_qubits >= 3]
                if real_backends:
                    self.backend = sorted(real_backends, key=lambda b: b.status().pending_jobs)[0]
                else:
                    raise RuntimeError("No hay backends cuánticos reales disponibles con ≥ 3 qubits.")
            
            print(f"🔌 Backend seleccionado: {self.backend.name} (Trabajos pendientes: {self.backend.status().pending_jobs})")

        except IBMRuntimeError as e:
            print(f"❌ Error al conectar con IBM Quantum Platform: {e}")
            self.service = None
            self.backend = None
        except Exception as e:
            print(f"❌ Ocurrió un error inesperado durante la conexión: {e}")
            self.service = None
            self.backend = None

    def mary(self, circ, angle, t, c0, c1):
      """Implementación de una CCRY-Gate usando CX y RY."""
      circ.ry(angle / 4, t)
      circ.cx(c0, t)
      circ.ry(-angle / 4, t)
      circ.cx(c1, t)
      circ.ry(angle / 4, t)
      circ.cx(c0, t)
      circ.ry(-angle / 4, t)
      circ.cx(c1, t)

    def apply_global_mary(self, qc, qr):
      """Aplica la puerta MARY para cada estado de posición."""
      # Posición |00>: pos1=0, pos0=0
      qc.x(qr[1])
      qc.x(qr[2])
      self.mary(qc, self.thetas[0], qr[0], qr[1], qr[2]) # Control en q1 y q2
      qc.x(qr[2])
      qc.x(qr[1])
      qc.barrier()

      # Posición |01>: pos1=0, pos0=1
      qc.x(qr[1])
      self.mary(qc, self.thetas[1], qr[0], qr[1], qr[2])
      qc.x(qr[1])
      qc.barrier()

      # Posición |10>: pos1=1, pos0=0
      qc.x(qr[2])
      self.mary(qc, self.thetas[2], qr[0], qr[1], qr[2])
      qc.x(qr[2])
      qc.barrier()

      # Posición |11>: pos1=1, pos0=1
      self.mary(qc, self.thetas[3], qr[0], qr[1], qr[2])
      qc.barrier()

    def build_circuit(self):
        """
        ## CORRECCIÓN ##: Se eliminó 'self.' al llamar a las clases de Qiskit.
        """
        qr = QuantumRegister(3, name='q')
        cr = ClassicalRegister(3, name='c')
        qc = QuantumCircuit(qr, cr, name="FRQI_MARY")

        qc.h(qr[1])
        qc.h(qr[2])
        qc.barrier()

        # Aplicar la codificación de la imagen
        self.apply_global_mary(qc, qr)

        qc.measure(qr, cr)
        self.qc = qc

    def run(self):
        """Ejecuta el circuito en el backend configurado."""
        self.build_circuit()
        self.counts = None

        if self.backend and self.service:
            print(f"\nEjecutando en el backend de IBM: {self.backend.name} con {self.shots} shots...")
            try:
                sampler = Sampler(mode=self.backend)
                transpiled_qc = transpile(self.qc, self.backend)
                
                # El método run espera una LISTA de circuitos.
                job = sampler.run([transpiled_qc], shots=self.shots)
                print(f"Job ID: {job.job_id()} enviado a {self.backend.name}.")
                print("Esperando resultados...")
                
                result = job.result()
                
                pub_result = result[0]
                self.counts = pub_result.data.c.get_counts()

                print("✅ Resultados obtenidos del backend de IBM:")
                print(self.counts)

            except Exception as e:
                print(f"❌ Error al ejecutar en el backend de IBM: {e}")
                print("⚠️ Volviendo al simulador local.")
                self._run_local_simulator()
        else:
            print("\nBackend de IBM no disponible. Ejecutando en el simulador QASM local...")
            self._run_local_simulator()

    def _run_local_simulator(self):
        """Ejecuta el circuito en el simulador local."""
        try:
            simulator = Aer.get_backend('qasm_simulator')
            transpiled_qc = transpile(self.qc, simulator)
            job = simulator.run(transpiled_qc, shots=self.shots)
            result = job.result()
            self.counts = result.get_counts(transpiled_qc) 
            print("✅ Resultados obtenidos del simulador QASM local:")
            print(self.counts)
        except Exception as e:
            print(f"❌ Error al ejecutar en el simulador local: {e}")
            self.counts = {}

    def analyze_results(self):
        """
        ## CORRECCIÓN ##: Se ajustó la lógica de parseo de bits para que sea
        robusta y consistente con la ejecución anterior.
        """
        if self.counts is None:
            raise RuntimeError("Primero debes ejecutar run() para obtener resultados.")

        print(f"\nAnalizando {self.shots} simulaciones...")
        self.results = {}
        # El orden de píxeles corresponde a la posición |q1q2>
        positions_order = ['00', '01', '10', '11'] 

        for i, target_pos_str in enumerate(positions_order):
            filtered_counts = {'0': 0, '1': 0}
            
            for bitstring, count in self.counts.items():
                # El bitstring es 'c2c1c0'. c0 es el qubit de color q0.
                color_bit = bitstring[-1]
                # La posición la dan q1 y q2 (c1 y c2).
                current_pos_str = bitstring[1] + bitstring[0]

                if current_pos_str == target_pos_str:
                    filtered_counts[color_bit] += count

            total = sum(filtered_counts.values())
            print(f"\nPosición |{target_pos_str}⟩ (de q1q2):")

            if total > 0:
                p0 = filtered_counts.get('0', 0) / total
                p1 = filtered_counts.get('1', 0) / total
            else:
                p0, p1 = 0.0, 0.0
                
            reconstructed_intensity_norm = np.sqrt(p1)
            reconstructed_intensity = reconstructed_intensity_norm * self.max_intensity

            self.results[target_pos_str] = {
                'p0': p0,
                'p1': p1,
                'reconstructed_intensity': int(round(reconstructed_intensity))
            }

            print(f"   Probabilidad de medir |0⟩ en el qubit de color: {p0:.4f}")
            print(f"   Probabilidad de medir |1⟩ en el qubit de color: {p1:.4f}")
            print(f"   Intensidad reconstruida: {reconstructed_intensity:.2f}")
            
    # El resto de métodos de ploteo y visualización no requieren cambios
    # y son compatibles con las correcciones hechas.
    
    def plot_histogram(self):
        if self.counts is None:
            raise RuntimeError("Primero debes ejecutar run() para obtener resultados.")
        plt.figure(figsize=(12, 7))
        plot_histogram(self.counts, title='Resultados de la Medición en Backend')
        plt.show()

    def plot_probabilities(self):
        if self.counts is None:
            raise RuntimeError("Primero debes ejecutar run() para obtener resultados.")
        
        total_shots = sum(self.counts.values())
        if total_shots == 0:
            print("No hay cuentas para graficar probabilidades.")
            return
            
        probabilities = {k: v / total_shots for k, v in self.counts.items()}
        
        plt.figure(figsize=(12, 7))
        ax = sns.barplot(x=list(probabilities.keys()), y=list(probabilities.values()))

        for container in ax.containers:
            ax.bar_label(container, fmt='%.4f', label_type='edge')

        plt.title('Probabilidades de Medición por Estado')
        plt.ylabel('Probabilidad')
        plt.xlabel('Estado Medido (q2q1q0)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def reconstruct_image(self):
        if not self.results:
            raise RuntimeError("Primero debes ejecutar analyze_results() para obtener resultados.")

        print("\n--- Reconstrucción de la Imagen ---")
        reconstructed_pixels = np.zeros((2, 2), dtype=int)

        reconstructed_pixels[0, 0] = self.results.get('00', {}).get('reconstructed_intensity', 0)
        reconstructed_pixels[0, 1] = self.results.get('01', {}).get('reconstructed_intensity', 0)
        reconstructed_pixels[1, 0] = self.results.get('10', {}).get('reconstructed_intensity', 0)
        reconstructed_pixels[1, 1] = self.results.get('11', {}).get('reconstructed_intensity', 0)

        original_pixels_display = np.array(self.intensities).reshape(2, 2)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        sns.heatmap(original_pixels_display, annot=True, fmt=".0f", cmap='gray', vmin=0, vmax=self.max_intensity, ax=axes[0], cbar=False, annot_kws={"color": "red"})
        axes[0].set_title('Imagen Original (Input)')
        
        sns.heatmap(reconstructed_pixels, annot=True, fmt=".0f", cmap='gray', vmin=0, vmax=self.max_intensity, ax=axes[1], cbar=False, annot_kws={"color": "red"})
        axes[1].set_title('Imagen Reconstruida (Output)')
        
        plt.tight_layout()
        plt.show()

    def print_state_table(self):
        positions = ['00', '01', '10', '11']
        
        intensity_norm = [i / self.max_intensity for i in self.intensities]
        cos_vals = [round(np.sqrt(1 - i), 5) for i in intensity_norm]
        sin_vals = [round(np.sqrt(i), 5) for i in intensity_norm]
        
        data = {
            'Posición |ij>': [f'$|{p}>$' for p in positions],
            'Intensidad': [f'{i:.0f}' for i in self.intensities],
            'Int. Norm. ($I_n$)': [f'{i_n:.3f}' for i_n in intensity_norm],
            '$\sqrt{1-I_n}$': [f'{c:.5f}' for c in cos_vals],
            '$\sqrt{I_n}$': [f'{s:.5f}' for s in sin_vals],
            'Estado de Color Esperado': [
                f"$({cos_vals[i]}|0> + {sin_vals[i]}|1>)$"
                for i in range(4)
            ]
        }
        df = pd.DataFrame(data)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print("\nTabla de Estados Esperados (Ideal):")
        print(df.to_string(index=False))

    def draw_circuit(self, output='mpl', filename=None):
        if self.qc is None:
            self.build_circuit()
        from qiskit.visualization import circuit_drawer
        print(f"\n--- Dibujando el Circuito Cuántico ---")
        display(circuit_drawer(self.qc, output=output, idle_wires=False))