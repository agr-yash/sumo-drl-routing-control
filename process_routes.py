import xml.etree.ElementTree as ET
import random
import argparse
import sys

def add_vehicle_types(input_file, output_file, av_mpr):
    """
    Reads a SUMO route file, adds AV and HV vehicle types, and assigns a type
    to each vehicle based on the Market Penetration Rate (MPR).
    """
    try:
        if not (0.0 <= av_mpr <= 1.0):
            raise ValueError("MPR must be a value between 0.0 and 1.0.")

        tree = ET.parse(input_file)
        root = tree.getroot()

        # --- THIS IS THE CRUCIAL PART THAT WAS MISSING FROM YOUR OUTPUT ---
        # It adds the <vType> definitions directly to the <routes> element.
        
        # Autonomous Vehicle (AV)
        ET.SubElement(root, 'vType', attrib={
            'id': 'AV', 'vClass': 'passenger', 'maxSpeed': '15',
            'carFollowModel': 'IDM', 'accel': '5.0', 'decel': '5.0'
        })

        # Human-driven Vehicle (HV)
        ET.SubElement(root, 'vType', attrib={
            'id': 'HV', 'vClass': 'passenger', 'maxSpeed': '15',
            'carFollowModel': 'IDM', 'accel': '5.0', 'decel': '5.0', 'tau': '1.5'
        })
        # --- END OF CRUCIAL PART ---

        vehicle_count = 0
        av_count = 0
        hv_count = 0
        
        for vehicle in root.findall('vehicle'):
            vehicle_count += 1
            if random.random() < av_mpr:
                vehicle.set('type', 'AV')
                av_count += 1
            else:
                vehicle.set('type', 'HV')
                hv_count += 1
        
        if sys.version_info >= (3, 9):
            ET.indent(tree)
        
        tree.write(output_file, encoding='UTF-8', xml_declaration=True)

        print("Success!")
        print(f"Processed {vehicle_count} vehicles.")
        print(f"Assigned {av_count} AVs and {hv_count} HVs.")
        print(f"New route file saved to: {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process SUMO routes for mixed traffic.")
    parser.add_argument("-i", "--input", default="city2.rou.xml", help="Input route file.")
    parser.add_argument("-o", "--output", default="city2.mixed.rou.xml", help="Output route file.")
    parser.add_argument("--mpr", type=float, default=0.5, help="AV Market Penetration Rate (0.0 to 1.0).")
    
    args = parser.parse_args()
    add_vehicle_types(args.input, args.output, args.mpr)