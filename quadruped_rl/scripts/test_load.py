import genesis as gs

def main():
    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        show_viewer = True,
        viewer_options = gs.options.ViewerOptions(
            camera_pos    = (2.0, -2.0, 1.5),
            camera_lookat = (0.0, 0.0, 0.5),
            camera_fov    = 40,
        ),
    )

    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    robot = scene.add_entity(
        morph=gs.morphs.MJCF(file='assets/xml_test.xml'),
    )

    scene.build()

    while True:
        scene.step()

if __name__ == "__main__":
    main()