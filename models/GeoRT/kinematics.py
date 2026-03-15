import pytorch_kinematics as pk

# Load the chain from the URDF file
chain = pk.build_chain_from_urdf(open("models/GeoRT/allegro_hand_description_right.urdf").read())

# chain = chain.to(device="mps")

print(chain.get_joint_parameter_names())
print()

transforms = chain.forward_kinematics([[0] * 16])
# thumbtip_pos = transforms['link_2.0'].get_matrix()[:, :3, 3]
for item in transforms:
    print(item)
    print(transforms[item])
    print()


print("done")