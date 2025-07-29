# Importaciones de librerías comunes
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

# Importaciones de Qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import RYGate
from qiskit.visualization import plot_histogram as qiskit_plot_histogram
from qiskit.visualization import circuit_drawer

# Importaciones para IBM Quantum
from qiskit_ibm_runtime.exceptions import IBMRuntimeError
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Sessio

class FRQI_MCRY_Simulator:
    """
    Simulador del algoritmo FRQI (Flexible Representation of Quantum Images).

    Esta clase permite codificar una imagen (representada por intensidades de píxeles)
    en un circuito cuántico utilizando la codificación FRQI, ejecutar el circuito
    en un simulador local o en una computadora cuántica real de IBM, y analizar
    los resultados.
    """

    def __init__(self, intensities=None, max_intensity=255, shots=8000, n=2):
        """
        Inicializa el simulador FRQI.

        Args:
            intensities (list, optional): Lista de intensidades de píxeles (0-255).
                                         Por defecto, [15, 25, 150, 255].
            max_intensity (int, optional): Valor máximo de intensidad para normalización.
                                           Por defecto, 255.
            shots (int, optional): Número de veces que se ejecuta el circuito cuántico.
                                   Por defecto, 80000.
        """
        # Atributos para IBM Quantum
        self.service = None # Instancia de QiskitRuntimeService
        self.backend = None # El backend a usar (simulador o real) - ahora almacenará el NOMBRE del backend
                            # ya que Sampler toma el nombre, no un objeto backend
        self.backend_object = None # This will store the actual backend object if retrieved for properties/status

        # Variables para correr el código
        self.n = n
        self.max_intensity = max_intensity
        self.shots = shots
        self.intensities = intensities if intensities is not None else [15, 25, 150, 255]

        # Calcular thetas desde intensidades
        self.thetas = self.calculate_thetas(self.intensities)

        self.results = {}
        self.probabilities_data = {}
        self.counts = None
        self.qc = None

    def calculate_thetas(self, intensities):
        """
        Calcula los ángulos theta para la codificación FRQI a partir de las intensidades.

        Args:
            intensities (list): Lista de intensidades de píxeles.

        Returns:
            list: Lista de ángulos theta en radianes.
        """
        # Normaliza intensidades al rango [0, 1]
        intensity_norm = [i / self.max_intensity for i in intensities]
        intensity_norm = [min(1.0, max(-1.0, i)) for i in intensity_norm]
        # Calcula thetas con la fórmula FRQI: theta = 2 * arcsin(intensidad_normalizada)
        thetas = [2 * np.arcsin(i) for i in intensity_norm]
        return thetas

    def connect_to_ibm_backend(self, token: str, backend_name: str = None, channel: str = "ibm_cloud", instance: str = None):
        """
        Conecta a una computadora cuántica real de IBM o a un simulador de IBM Quantum
        utilizando Qiskit Runtime.

        Args:
            token (str): Tu token de API de IBM Quantum Experience.
            backend_name (str, optional): Nombre específico del backend de IBM Quantum
                                         al que deseas conectarte (ej. 'ibm_lagos').
                                         Si es None, se buscará el backend cuántico real
                                         menos ocupado.
            channel (str, optional): El canal para conectarse, 'ibm_cloud' o 'ibm_quantum'.
                                    'ibm_cloud' para IBM Quantum Platform (pay-as-you-go).
                                    'ibm_quantum' para la experiencia IBM Quantum (anterior, sin coste).
                                    Por defecto, 'ibm_cloud'.
            instance (str, optional): La instancia de servicio (hub/group/project) para cuentas
                                      de IBM Quantum Platform. Obligatorio si channel es 'ibm_cloud'.
                                      Ej: "ibm-q/open/main".
        """
        try:
            # Inicializa el servicio de Qiskit Runtime
            if channel == "ibm_cloud" and instance:
                self.service = QiskitRuntimeService(channel=channel, token=token, instance=instance)
                print(f"Conectado a IBM Quantum Platform con instancia: {instance}.")
            elif channel == "ibm_quantum":
                self.service = QiskitRuntimeService(channel=channel, token=token)
                print("Conectado a la experiencia clásica de IBM Quantum (si las credenciales están guardadas localmente).")
            else:
                print("Error: 'instance' es obligatorio para 'ibm_cloud' channel. O especifica 'ibm_quantum' como channel.")
                self.service = None
                self.backend = None
                return

            if backend_name:
                # Store the name of the backend. Sampler will take the name.
                self.backend = backend_name
                # Optionally, get the backend object to check its properties (e.g., status, pending jobs)
                # This is useful for user feedback, but Sampler doesn't strictly need the object.
                try:
                    self.backend_object = self.service.backend(backend_name)
                    print(f"Backend seleccionado: {self.backend_object.name}")
                except Exception as e:
                    print(f"Advertencia: No se pudo obtener el objeto para el backend '{backend_name}'. Puede que no exista o no esté disponible. Error: {e}")
                    self.backend_object = None
            else:
                # If no specific backend_name is provided, find the least busy *real* quantum computer.
                # We need to get the list of backends first, then filter and sort.
                # Note: self.service.backends() returns a list of backend *names*.
                # To get backend *objects* for properties like 'status().pending_jobs',
                # you'd iterate and call service.backend(name).

                # For Sampler, we just need the name.
                # Let's list and find the least busy *name*

                # First, get all available backends (names)
                all_backend_names = self.service.backends()

                # Filter for non-simulators, operational, and sufficient qubits
                available_real_backends = []
                for name in all_backend_names:
                    try:
                        # Get the full backend object to check properties
                        b = self.service.backend(name)
                        if b.num_qubits >= 3 and not b.simulator and b.operational:
                            available_real_backends.append(b)
                    except Exception:
                        # Ignore backends that might not be accessible or have issues
                        pass

                if not available_real_backends:
                    print("No se encontraron computadoras cuánticas reales operativas con al menos 3 qubits. Se utilizará el simulador local por defecto.")
                    self.backend = None # Force local simulator usage
                    self.backend_object = None
                    return

                # Sort by pending jobs and pick the least busy
                least_busy_backend_object = sorted(available_real_backends, key=lambda b: b.status().pending_jobs)[0]
                self.backend = least_busy_backend_object.name
                self.backend_object = least_busy_backend_object # Store the object for potential future use (e.g., plotting)
                print(f"Backend cuántico real menos ocupado seleccionado: {self.backend_object.name}")

        except IBMRuntimeError as e:
            print(f"Error al conectar con IBM Quantum: {e}")
            print("Asegúrate de que tu token de API es válido, tu instancia (instance) es correcta si la usas, y tienes conexión a internet.")
            self.service = None
            self.backend = None
            self.backend_object = None
        except Exception as e:
            print(f"Ocurrió un error inesperado durante la conexión: {e}")
            self.service = None
            self.backend = None
            self.backend_object = None


    def build_circuit(self):
        """
        Construye el circuito cuántico FRQI.

        El circuito utiliza 3 qubits:
        - q[0]: qubit de color (intensidad)
        - q[1], q[2]: qubits de posición (para 4 píxeles, 2x2 imagen)
        """
        # Crear registros cuánticos y clásicos
        qr = QuantumRegister(3, name='q')
        cr = ClassicalRegister(3, name='c')
        qc = QuantumCircuit(qr, cr, name="FRQI_MRCY")

        # Aplicar Hadamard a los qubits de posición para crear superposición
        qc.h(qr[1])  # pos1 (q[1] es el MSB de la posición)
        qc.h(qr[2])  # pos0 (q[2] es el LSB de la posición)

        # Los qubits de control son q[1] (más significativo) y q[2] (menos significativo)
        control_qubits = [qr[1], qr[2]]
        # El qubit objetivo es q[0]
        target_qubit = qr[0]

        # Iterar sobre cada píxel (posición) y aplicar la rotación RY controlada
        for i, theta in enumerate(self.thetas):

            bin_str = format(i, '02b')
            control_state = bin_str[0] + bin_str[1] # pos1_bit + pos0_bit

            if bin_str[0] == '0':
                qc.x(qr[1]) # Invertir q[1] si su bit de control es '0'
            if bin_str[1] == '0':
                qc.x(qr[2]) # Invertir q[2] si su bit de control es '0'

            mary_gate = RYGate(theta).control(len(control_qubits))
            qc.append(mary_gate, control_qubits + [target_qubit])


            # Deshacer las puertas X aplicadas para la selección de posición
            if bin_str[0] == '0':
                qc.x(qr[1])
            if bin_str[1] == '0':
                qc.x(qr[2])

        # Medir todos los qubits y almacenar los resultados en los registros clásicos
        qc.measure(qr, cr)
        self.qc = qc


    def run(self):
        """
        Ejecuta el circuito cuántico FRQI en el backend seleccionado.
        Si se ha conectado a un backend de IBM Quantum, lo usa; de lo contrario,
        usa el simulador QASM local.
        """
        self.build_circuit() # Primero construye el circuito

        # Verificar si un nombre de backend (self.backend) y el servicio están inicializados
        if self.backend and self.service:
            print(f"\nEjecutando en el backend de IBM Quantum: {self.backend} con {self.shots} shots (usando Sampler)...")
            try:
                # 1. Inicializa SamplerV2 SIN ARGUMENTOS en el constructor.
                # Esto es lo que el error 'unexpected keyword argument service' nos ha dicho.
                sampler = Sampler() # <--- ¡Aquí está la clave! Constructor vacío.

                # 2. Pasa el circuito, los shots, el backend Y el servicio al método .run().
                # Esto es lo que el error 'A backend or session must be specified.' nos ha dicho.
                job = sampler.run(
                    circuits=self.qc,  # El circuito (o lista de circuitos)
                    shots=self.shots,  # El número de mediciones
                    backend=self.backend, # El nombre del backend (string)
                    service=self.service # ¡Tu instancia QiskitRuntimeService!
                )

                print(f"Trabajo enviado. ID del trabajo: {job.job_id}")
                print("Esperando resultados... Esto puede tomar un tiempo (puedes verificar el estado en https://quantum.ibm.com/jobs).")

                result = job.result()
                # SamplerV2 devuelve los resultados en un formato ligeramente diferente.
                # result[0] es para el primer (y único) circuito en la lista.
                # .data.meas.get_counts() es la forma correcta de obtener las cuentas para SamplerV2.
                self.counts = result[0].data.meas.get_counts()

                print("Resultados obtenidos del backend de IBM Quantum:")
                print(self.counts)
            except Exception as e:
                print(f"Error al ejecutar en el backend de IBM Quantum: {e}")
                print("Volviendo al simulador local.")
                self.backend = None # Reiniciar backend para usar simulador local
                self.backend_object = None # Reiniciar también el objeto backend
                self._run_local_simulator() # Reintentar con simulador local
        else:
            # Si no hay backend de IBM Quantum o hubo un error, usa el simulador local
            print("\nEjecutando en el simulador QASM local...")
            self._run_local_simulator()

    def _run_local_simulator(self):
        """Método interno para ejecutar en el simulador QASM local."""
        sim = Aer.get_backend('qasm_simulator')
        # For local simulators, you often need to transpile yourself if you want to optimize for a specific device model.
        # But for basic qasm_simulator, direct execution is fine.
        compiled = transpile(self.qc, sim) # Still good practice for consistency
        result = sim.run(compiled, shots=self.shots).result()
        self.counts = result.get_counts(self.qc)
        print("Resultados obtenidos del simulador local:")
        print(self.counts)

    def analyze_results(self):
        """
        Analiza los resultados de las mediciones del circuito para extraer
        las probabilidades de color para cada posición y reconstruir la intensidad.
        """
        if self.counts is None:
            raise RuntimeError("Primero debes ejecutar run() para obtener resultados.")

        print(f"\nCantidad de simulaciones: {self.shots}")
        print("\nMediciones del qubit de color q[0] para cada posición:")
        self.results = {}
        self.plot_probabilities_data = {} # Reinicializar aquí

        positions_order = ['00', '01', '10', '11'] # Orden de las intensidades originales
        prob = []

        for i, target_pos_str in enumerate(positions_order):

            filtered_counts = {'0': 0, '1': 0} # Inicializar conteos para '0' y '1' del qubit de color
            #print(filtered_counts)

            for bitstring, count in self.counts.items():
                color_bit = bitstring[2] # q[0]
                pos1_bit = bitstring[1] # q[1]
                pos0_bit = bitstring[0] # q[2]

                current_pos_str = pos1_bit + pos0_bit

                if current_pos_str == target_pos_str:
                    filtered_counts[color_bit] += count

            total = sum(filtered_counts.values()) # Conteo total para esta posición
            print(f"\nPosición |{target_pos_str}> (de q[1]q[2] medido):")
            #print(filtered_counts.values())
            if total > 0:
                p0 = filtered_counts.get('0', 0) / total
                #print(f"p0: {p0}")
                p1 = filtered_counts.get('1', 0) / total
                #print(f"p1: {p1}")
            else:
                p0, p1 = 0.0, 0.0

            # --- CORRECCIÓN CLAVE AQUÍ ---
            # Para FRQI, la probabilidad de medir |1> es (normalized_intensity)^2
            # Por lo tanto, la intensidad normalizada reconstruida es la raíz cuadrada de P(1).

            reconstructed_intensity_norm = np.sqrt(p1)
            reconstructed_intensity = reconstructed_intensity_norm * self.max_intensity

            self.results[target_pos_str] = {
                '0': p0,
                '1': p1,
                'reconstructed_intensity': reconstructed_intensity
            }

            self.plot_probabilities_data[f'|{target_pos_str}> (P(0))'] = p0
            self.plot_probabilities_data[f'|{target_pos_str}> (P(1))'] = p1

            print(f"   Posición |{target_pos_str}⟩ (de q[1]q[2] medido):")
            print(f"     Probabilidad de 0 (medido en q[2]): {p0:.4f}")
            print(f"     Probabilidad de 1 (medido en q[1]): {p1:.4f}")

    def plot_histogram(self):
        """
        Muestra un histograma de los conteos de medición del circuito.
        """
        if self.counts is None:
            raise RuntimeError("Primero debes ejecutar run() para obtener resultados.")

        counts_str_keys = {str(k): v for k, v in self.counts.items()}

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=list(counts_str_keys.keys()), y=list(counts_str_keys.values()))

        for container in ax.containers:
            ax.bar_label(container, fmt='%.0f', label_type='edge')

        plt.title('Valores de Medición por base computacional')
        plt.ylabel('Cuentas')
        plt.xlabel('Estado Medido (q[2]q[1]q[0])')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def plot_probabilities(self):
        """
        Genera un gráfico de barras mostrando las probabilidades de cada estado
        medido (q[2]q[1]q[0]).
        """
        if self.counts is None:
            raise RuntimeError("Primero debes ejecutar run() para obtener resultados.")

        probabilities = {k: v / self.shots for k, v in self.counts.items()}
        probabilities_str_keys = {str(k): v for k, v in probabilities.items()}

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=list(probabilities_str_keys.keys()), y=list(probabilities_str_keys.values()))

        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', label_type='edge')

        plt.title('Probabilidades de Medición por base computacional')
        plt.ylabel('Probabilidad')
        plt.xlabel('Estado Medido (q[2]q[1]q[0])')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def reconstruct_image(self):
        """
        Reconstruye y muestra la imagen en escala de grises a partir de
        las intensidades calculadas. Asume una imagen de 2x2 píxeles.
        """
        if not self.results:
            raise RuntimeError("Primero debes ejecutar analyze_results() para obtener resultados.")

        print("\n--- Reconstrucción de la Imagen ---")

        # Asumiendo un orden de píxeles para una imagen 2x2:
        # Píxel (0,0) -> Posición '00'
        # Píxel (0,1) -> Posición '01'
        # Píxel (1,0) -> Posición '10'
        # Píxel (1,1) -> Posición '11'

        # Crea una matriz para la imagen de 2x2
        reconstructed_pixels = np.zeros((2, 2), dtype=int)

        # Asigna los valores reconstruidos a la matriz en el orden correcto
        reconstructed_pixels[0, 0] = self.results['00']['reconstructed_intensity']
        reconstructed_pixels[0, 1] = self.results['01']['reconstructed_intensity']
        reconstructed_pixels[1, 0] = self.results['10']['reconstructed_intensity']
        reconstructed_pixels[1, 1] = self.results['11']['reconstructed_intensity']

        print("\nIntensidades Originales (Input):")
        original_pixels_display = np.array([
            [self.intensities[0], self.intensities[1]],
            [self.intensities[2], self.intensities[3]]
        ])
        #print(original_pixels_display)

        print("\nIntensidades Reconstruidas (Output):")
        #print(reconstructed_pixels)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        axes[0].imshow(original_pixels_display, cmap='gray', vmin=0, vmax=self.max_intensity)
        axes[0].set_title('Imagen Original (Input)')
        axes[0].set_xticks(np.arange(2))
        axes[0].set_yticks(np.arange(2))
        axes[0].set_xticklabels(['Col 0', 'Col 1'])
        axes[0].set_yticklabels(['Fila 0', 'Fila 1'])
        for (j, i), val in np.ndenumerate(original_pixels_display):
            axes[0].text(i, j, f'{val}', ha='center', va='center', color='red', fontsize=12)

        axes[1].imshow(reconstructed_pixels, cmap='gray', vmin=0, vmax=self.max_intensity)
        axes[1].set_title('Imagen Reconstruida (Output)')
        axes[1].set_xticks(np.arange(2))
        axes[1].set_yticks(np.arange(2))
        axes[1].set_xticklabels(['Col 0', 'Col 1'])
        axes[1].set_yticklabels(['Fila 0', 'Fila 1'])
        for (j, i), val in np.ndenumerate(reconstructed_pixels):
            axes[1].text(i, j, f'{val}', ha='center', va='center', color='red', fontsize=12)

        plt.tight_layout()
        plt.show()

    def print_state_table(self):
        """
        Muestra una tabla con las intensidades normalizadas, ángulos theta,
        y los estados parciales esperados para cada posición.
        """
        positions = ['00', '01', '10', '11']
        cos_vals = [round(np.cos(theta), 5) for theta in self.thetas]
        sin_vals = [round(np.sin(theta), 5) for theta in self.thetas]

        data = {
            'Posición $|i>$': [f'$|{p}>$' for p in positions],
            'Intensidad Normalizada': [f'{i:.0f}' for i in self.intensities],
            f'theta_i (rad)': [f'{theta:.5f}' for theta in self.thetas],
            'cos(theta_i)': [f'{c:.5f}' for c in cos_vals],
            'sin(theta_i)': [f'{s:.5f}' for s in sin_vals],
            'Estado Parcial Esperado': [
                f"$({cos_vals[i]}|0> + {sin_vals[i]}|1>)|{positions[i]}>$"
                for i in range(4)
            ]
        }

        df = pd.DataFrame(data)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print("\nTabla de Estados Esperados (Ideal):")
        print(df.to_string(index=False))

    def draw_circuit(self, output='mpl', filename=None, style=None):
        """
        Dibuja y muestra el circuito cuántico.

        Args:
            output (str, optional): Formato de salida del dibujo.
                                    Puede ser 'text', 'mpl' (matplotlib), 'latex', etc.
                                    Por defecto, 'mpl'.
            filename (str, optional): Nombre del archivo para guardar la imagen si output es 'mpl' o 'latex'.
                                      Si es None, se muestra en pantalla.
            style (dict, optional): Un diccionario de estilos a pasar al dibujante del circuito.
                                    Por defecto, None.
                                    **Nota:** Si 'mpl' es el output, se usa el estilo "clifford" para evitar la advertencia.
        """
        if self.qc is None:
            self.build_circuit()

        print(f"\n--- Dibujando el Circuito Cuántico ({output}) ---")
        if output == 'mpl':
            fig = circuit_drawer(self.qc, output=output, idle_wires=False, style="clifford")
            if filename:
                fig.savefig(filename, bbox_inches='tight')
                print(f"Circuito guardado como '{filename}'")
                plt.close(fig)
            else:
                display(fig)
                plt.close(fig)
        elif output == 'text':
            print(circuit_drawer(self.qc, output=output, idle_wires=False))
        else:
            try:
                result = circuit_drawer(self.qc, output=output, filename=filename, idle_wires=False, style=style)
                if filename:
                    print(f"Circuito guardado como '{filename}'")
                elif isinstance(result, str):
                    print(result)
                elif result is not None:
                    display(result)
            except Exception as e:
                print(f"No se pudo dibujar el circuito con el formato '{output}'. Error: {e}")
                print("Intenta instalar las dependencias necesarias, por ejemplo: pip install pylatexenc qiskit-terra[visualization]")