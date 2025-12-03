# 13.5 Outils de Développement

---

## Introduction

Cette section présente les **outils essentiels** pour le développement FPGA, des environnements de développement intégrés (IDE) aux outils de simulation et de debug.

---

## Écosystème d'Outils

```
┌─────────────────────────────────────────────────────────────────┐
│                    FPGA Development Tools                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         Design Entry                                      │  │
│  │  • Text Editors (VS Code, Vim)                          │  │
│  │  • HDL Plugins                                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         Simulation                                        │  │
│  │  • ModelSim / Questa                                     │  │
│  │  • Vivado Simulator                                      │  │
│  │  • Verilator (open source)                              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         Synthesis & Implementation                        │  │
│  │  • Vivado (Xilinx)                                       │  │
│  │  • Quartus (Intel)                                       │  │
│  │  • Yosys (open source synthesis)                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         HLS                                               │  │
│  │  • Vitis HLS                                             │  │
│  │  • Intel HLS Compiler                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         Debug & Analysis                                  │  │
│  │  • Signal Tap / ILA                                      │  │
│  │  • Timing Analyzer                                       │  │
│  │  • Power Analyzer                                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Vivado Design Suite (Xilinx)

```python
class VivadoSuite:
    """
    Suite d'outils Vivado de Xilinx
    """
    
    components = {
        'vivado': {
            'name': 'Vivado Design Suite',
            'purpose': 'Synthesis, Implementation, Bitstream',
            'features': [
                'Design entry (HDL, IP)',
                'Synthesis (Verilog, VHDL)',
                'Implementation (Place & Route)',
                'Timing analysis',
                'Power analysis',
                'Bitstream generation'
            ],
            'gui': 'Tcl-based GUI',
            'scripting': 'Tcl scripting support'
        },
        'vitis_hls': {
            'name': 'Vitis HLS',
            'purpose': 'High-Level Synthesis',
            'features': [
                'C/C++ to RTL',
                'C simulation',
                'C/RTL co-simulation',
                'IP export'
            ]
        },
        'vivado_simulator': {
            'name': 'Vivado Simulator',
            'purpose': 'HDL Simulation',
            'features': [
                'Verilog/VHDL simulation',
                'Mixed-language support',
                'Waveform viewer'
            ]
        },
        'vivado_hls': {
            'name': 'Vivado HLS (Legacy)',
            'note': 'Remplacé par Vitis HLS dans versions récentes'
        }
    }
    
    @staticmethod
    def display_components():
        """Affiche les composants Vivado"""
        print("\n" + "="*60)
        print("Vivado Design Suite Components")
        print("="*60)
        
        for comp_id, comp_info in VivadoSuite.components.items():
            print(f"\n{comp_info['name']} ({comp_id}):")
            print(f"  Purpose: {comp_info['purpose']}")
            print(f"  Features:")
            for feature in comp_info['features']:
                print(f"    • {feature}")
            if 'note' in comp_info:
                print(f"  Note: {comp_info['note']}")

VivadoSuite.display_components()
```

---

## Workflow Vivado

```python
class VivadoWorkflow:
    """
    Workflow typique avec Vivado
    """
    
    def generate_tcl_script(self):
        """Génère un script Tcl typique pour Vivado"""
        tcl_script = """
# Vivado Tcl Script Example

# Create project
create_project my_project ./my_project -part xc7z020clg400-1

# Add source files
add_files {./src/top_module.v}
add_files {./src/adder.v}

# Add constraints
add_files -fileset constrs_1 {./constraints/timing.xdc}
add_files -fileset constrs_1 {./constraints/pinout.xdc}

# Run synthesis
launch_runs synth_1
wait_on_run synth_1

# Check synthesis results
open_run synth_1
report_utilization -file utilization.rpt
report_timing_summary -file timing.rpt

# Run implementation
launch_runs impl_1 -jobs 4
wait_on_run impl_1

# Check implementation
open_run impl_1
report_utilization -file impl_utilization.rpt
report_timing_summary -file impl_timing.rpt
report_power -file power.rpt

# Generate bitstream
launch_runs impl_1 -to_step write_bitstream
wait_on_run impl_1
"""
        return tcl_script
    
    def display_workflow_steps(self):
        """Affiche les étapes du workflow"""
        steps = [
            '1. Create project (GUI or Tcl)',
            '2. Add source files (HDL, IP cores)',
            '3. Add constraints (.xdc files)',
            '4. Run synthesis',
            '5. Review synthesis reports',
            '6. Run implementation (Place & Route)',
            '7. Review timing/power/utilization',
            '8. Generate bitstream',
            '9. Program device'
        ]
        
        print("\n" + "="*60)
        print("Vivado Workflow Steps")
        print("="*60)
        for step in steps:
            print(f"  {step}")

workflow = VivadoWorkflow()
workflow.display_workflow_steps()

print("\n" + "="*60)
print("Example Tcl Script")
print("="*60)
print(workflow.generate_tcl_script())
```

---

## Simulation Tools

```python
class SimulationTools:
    """
    Outils de simulation HDL
    """
    
    tools = {
        'modelsim': {
            'vendor': 'Mentor Graphics (Siemens)',
            'languages': ['Verilog', 'VHDL', 'SystemVerilog'],
            'features': [
                'Mixed-language simulation',
                'Advanced debugging',
                'Code coverage',
                'Assertion-based verification'
            ],
            'license': 'Commercial (free student edition available)'
        },
        'questa': {
            'vendor': 'Mentor Graphics (Siemens)',
            'description': 'Enterprise version of ModelSim',
            'features': [
                'All ModelSim features',
                'Advanced UVM support',
                'Performance optimization'
            ]
        },
        'vivado_simulator': {
            'vendor': 'Xilinx',
            'languages': ['Verilog', 'VHDL'],
            'features': [
                'Integrated with Vivado',
                'Free with Vivado',
                'Good for Xilinx-specific simulation'
            ]
        },
        'verilator': {
            'vendor': 'Open source',
            'languages': ['Verilog', 'SystemVerilog'],
            'features': [
                'Fast compilation',
                'C++ simulation backend',
                'Free and open source',
                'Popular for CI/CD'
            ],
            'limitations': 'VHDL not supported'
        },
        'iverilog': {
            'vendor': 'Open source',
            'languages': ['Verilog'],
            'features': [
                'Free and open source',
                'Lightweight',
                'Good for simple designs'
            ]
        }
    }
    
    @staticmethod
    def display_tools():
        """Affiche les outils de simulation"""
        print("\n" + "="*60)
        print("HDL Simulation Tools")
        print("="*60)
        
        for tool_name, tool_info in SimulationTools.tools.items():
            print(f"\n{tool_name.replace('_', ' ').title()}:")
            for key, value in tool_info.items():
                if isinstance(value, list):
                    print(f"  {key}:")
                    for item in value:
                        print(f"    • {item}")
                else:
                    print(f"  {key}: {value}")

SimulationTools.display_tools()
```

---

## In-Circuit Debug

### Integrated Logic Analyzer (ILA)

```python
class ILA_Debug:
    """
    Integrated Logic Analyzer pour debug in-circuit
    """
    
    def __init__(self):
        self.concepts = {
            'ila': {
                'name': 'Integrated Logic Analyzer',
                'purpose': 'Debug in-circuit sans probes externes',
                'how': 'Inserté dans le design FPGA',
                'features': [
                    'Capture de signaux internes',
                    'Trigger conditions',
                ]
            },
            'signal_tap': {
                'name': 'Signal Tap Logic Analyzer',
                'vendor': 'Intel/Altera',
                'equivalent': 'ILA de Xilinx',
                'features': [
                    'Similaire à ILA',
                    'Intel-specific'
                ]
            }
        }
    
    def display_concepts(self):
        """Affiche les concepts de debug"""
        print("\n" + "="*60)
        print("In-Circuit Debug Tools")
        print("="*60)
        
        for concept, info in self.concepts.items():
            print(f"\n{info['name']}:")
            for key, value in info.items():
                if key != 'name':
                    if isinstance(value, list):
                        print(f"  {key}:")
                        for item in value:
                            print(f"    • {item}")
                    else:
                        print(f"  {key}: {value}")

# Exemple de configuration ILA dans Vivado
ILA_EXAMPLE = """
# Vivado Tcl pour créer un ILA

# Créer le debug core ILA
create_debug_core u_ila_0 ila
set_property C_DATA_DEPTH 1024 [get_debug_cores u_ila_0]
set_property C_TRIGIN_EN false [get_debug_cores u_ila_0]
set_property C_TRIGOUT_EN false [get_debug_cores u_ila_0]

# Ajouter des probes
set_property PROBE_TYPE DATA_AND_TRIGGER [get_debug_ports u_ila_0/probe0]
set_property port_width 8 [get_debug_ports u_ila_0/probe0]
connect_debug_port u_ila_0/probe0 [get_nets {data_signal[7:0]}]
"""

ila = ILA_Debug()
ila.display_concepts()

print("\n" + "="*60)
print("Example: ILA Configuration (Vivado Tcl)")
print("="*60)
print(ILA_EXAMPLE)
```

---

## Outils Open Source

```python
class OpenSourceTools:
    """
    Outils open source pour FPGA
    """
    
    tools = {
        'yosys': {
            'name': 'Yosys',
            'purpose': 'Verilog synthesis',
            'features': [
                'Synthesis de Verilog',
                'Technology mapping',
                'Scriptable avec Tcl',
                'Support plusieurs targets'
            ],
            'targets': ['Lattice iCE40', 'Xilinx (via prjxray)', 'Intel (via Trellis)']
        },
        'nextpnr': {
            'name': 'nextpnr',
            'purpose': 'Place & Route',
            'features': [
                'Placement et routing',
                'Timing-driven',
                'Open source'
            ],
            'targets': ['Lattice', 'Xilinx 7-series']
        },
        'verilator': {
            'name': 'Verilator',
            'purpose': 'Verilog simulator',
            'features': [
                'Compilation rapide',
                'C++ backend',
                'Très populaire'
            ]
        },
        'gtkwave': {
            'name': 'GTKWave',
            'purpose': 'Waveform viewer',
            'features': [
                'Visualise VCD/FST files',
                'Open source',
                'Multi-platform'
            ]
        },
        'icestorm': {
            'name': 'Project IceStorm',
            'purpose': 'Toolchain pour Lattice iCE40',
            'components': ['yosys', 'nextpnr', 'icepack']
        }
    }
    
    @staticmethod
    def display_tools():
        """Affiche les outils open source"""
        print("\n" + "="*60)
        print("Open Source FPGA Tools")
        print("="*60)
        
        for tool_id, tool_info in OpenSourceTools.tools.items():
            print(f"\n{tool_info['name']} ({tool_id}):")
            print(f"  Purpose: {tool_info['purpose']}")
            if 'features' in tool_info:
                print(f"  Features:")
                for feature in tool_info['features']:
                    print(f"    • {feature}")
            if 'targets' in tool_info:
                print(f"  Targets: {', '.join(tool_info['targets'])}")
            if 'components' in tool_info:
                print(f"  Components: {', '.join(tool_info['components'])}")

OpenSourceTools.display_tools()
```

---

## Scripts et Automatisation

```python
class AutomationScripts:
    """
    Scripts pour automatiser le workflow
    """
    
    def generate_makefile_example(self):
        """Génère un Makefile typique"""
        makefile = """
# Makefile for FPGA project

PROJECT = my_design
PART = xc7z020clg400-1

.PHONY: all synth impl bit clean

all: bit

synth:
	vivado -mode batch -source scripts/synth.tcl

impl:
	vivado -mode batch -source scripts/impl.tcl

bit: impl
	vivado -mode batch -source scripts/bitstream.tcl

sim:
	vivado -mode batch -source scripts/sim.tcl

clean:
	rm -rf *.jou *.log *.str *.runs .Xil

"""
        return makefile
    
    def generate_ci_example(self):
        """Génère un exemple CI/CD"""
        github_actions = """
# .github/workflows/fpga.yml
name: FPGA Build

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Install Vivado
        run: |
          # Download and install Vivado (requires license)
          
      - name: Run Synthesis
        run: |
          source /opt/Xilinx/Vivado/2020.1/settings64.sh
          make synth
          
      - name: Check Timing
        run: |
          python scripts/check_timing.py
"""
        return github_actions

automation = AutomationScripts()

print("\n" + "="*60)
print("Automation Examples")
print("="*60)

print("\nMakefile Example:")
print(automation.generate_makefile_example())

print("\n" + "="*60)
print("CI/CD Example (GitHub Actions)")
print("="*60)
print(automation.generate_ci_example())
```

---

## Exercices

### Exercice 13.5.1
Créez un script Tcl complet pour synthétiser et implémenter un design simple avec génération de rapports.

### Exercice 13.5.2
Configurez un ILA dans Vivado pour déboguer un signal interne d'un module.

---

## Points Clés à Retenir

> 📌 **Vivado: Suite complète pour Xilinx FPGA**

> 📌 **Simulation: ModelSim, Questa, Verilator (open source)**

> 📌 **ILA/Signal Tap: Debug in-circuit essentiel**

> 📌 **Outils open source: Yosys, nextpnr, Verilator**

> 📌 **Automatisation: Scripts Tcl, Makefiles, CI/CD**

> 📌 **Tcl scripting: Puissant pour automatisation Vivado**

---

*Chapitre suivant : [Chapitre 14 - Déploiement de Réseaux de Neurones sur FPGA](../Chapitre_14_NN_sur_FPGA/14_introduction.md)*

