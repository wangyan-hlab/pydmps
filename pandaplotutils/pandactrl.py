from panda3d.core import PerspectiveLens, AmbientLight, PointLight, Vec4, Vec3, Point3, WindowProperties, Filename, NodePath
from direct.filter.CommonFilters import CommonFilters
from direct.showbase.ShowBase import ShowBase
import pandaplotutils.inputmanager as im
import pandaplotutils.pandageom as pg
from panda3d.bullet import BulletWorld
from panda3d.bullet import BulletDebugNode
import os
import math
from utiltools.thirdparty import p3dhelper
from utiltools.thirdparty import o3dhelper
import utiltools.robotmath as rm

class World(ShowBase, object):

    def __init__(self, camp=[2000,500,2000], lookatpos=[0, 0, 250], up = [0, 0, 1], fov = 40, w = 2000, h = 1500, toggledebug = False, autocamrotate = False):
        """

        :param camp:
        :param lookatpos:
        :param fov:
        :param w: width of window
        :param h: height of window
        """

        # the taskMgr, loader, render2d, etc. are added to builtin after initializing the showbase parental class
        super().__init__()

        self.setBackgroundColor(1, 1, 1)

        # set up lens
        lens = PerspectiveLens()
        lens.setFov(fov)
        lens.setNearFar(1, 50000)
        # disable the default mouse control
        self.disableMouse()
        self.cam.setPos(camp[0], camp[1], camp[2])
        self.cam.lookAt(Point3(lookatpos[0], lookatpos[1], lookatpos[2]), Vec3(up[0], up[1], up[2]))
        self.cam.node().setLens(lens)

        # set up slight
        ablight = AmbientLight("ambientlight")
        ablight.setColor(Vec4(0.2, 0.2, 0.2, 1))
        self.__ablightnode = self.cam.attachNewNode(ablight)
        self.render.setLight(self.__ablightnode)

        ptlight0 = PointLight("pointlight1")
        ptlight0.setColor(Vec4(1, 1, 1, 1))
        self.__ptlightnode0 = self.cam.attachNewNode(ptlight0)
        self.__ptlightnode0.setPos(0, 0, 0)
        self.render.setLight(self.__ptlightnode0)

        ptlight1 = PointLight("pointlight1")
        ptlight1.setColor(Vec4(.4, .4, .4, 1))
        self.__ptlightnode1 = self.cam.attachNewNode(ptlight1)
        self.__ptlightnode1.setPos(self.cam.getPos().length(), 0, self.cam.getPos().length())
        self.render.setLight(self.__ptlightnode1)

        ptlight2 = PointLight("pointlight2")
        ptlight2.setColor(Vec4(.3, .3, .3, 1))
        self.__ptlightnode2 = self.cam.attachNewNode(ptlight2)
        self.__ptlightnode2.setPos(-self.cam.getPos().length(), 0, base.cam.getPos().length())
        self.render.setLight(self.__ptlightnode2)

        # helpers
        # use pg to access the util functions; use pggen to generate the geometries for decoration
        # for back-compatibility purpose
        self.pg = pg
        self.pggen = pg.PandaDecorator()
        self.p3dh = p3dhelper
        self.o3dh = o3dhelper
        self.rm = rm

        # set up inputmanager
        self.inputmgr = im.InputManager(self, lookatpos, self.pggen)
        taskMgr.add(self.__interactionUpdate, "interaction", appendTask=True)

        # set up rotational cam
        self.lookatp = lookatpos
        if autocamrotate:
            taskMgr.doMethodLater(.1, self.__rotatecamUpdate, "rotate cam")

        # set window size
        props = WindowProperties()
        props.setSize(w, h)
        self.win.requestProperties(props)

        # set up cartoon effect
        self.__separation = 1
        self.filters = CommonFilters(self.win, self.cam)
        self.filters.setCartoonInk(separation=self.__separation)
        # self.setCartoonShader(False)

        # set up physics world
        self.physicsworld = BulletWorld()
        self.physicsworld.setGravity(Vec3(0, 0, -981))
        taskMgr.add(self.__physicsUpdate, "physics", appendTask = True)
        if toggledebug:
            globalbprrender = base.render.attachNewNode("globalbpcollider")
            debugNode = BulletDebugNode('Debug')
            debugNode.showWireframe(True)
            debugNode.showConstraints(True)
            debugNode.showBoundingBoxes(False)
            debugNode.showNormals(True)
            self._debugNP = globalbprrender.attachNewNode(debugNode)
            self._debugNP.show()
            self.physicsworld.setDebugNode(self._debugNP.node())

        # set up render update
        self.__objtodraw = [] # the nodepath, collision model, or bullet dynamics model to be drawn
        taskMgr.add(self.__renderUpdate, "render", appendTask = True)

    def __interactionUpdate(self, task):
        # reset aspect ratio
        aspectRatio = self.getAspectRatio()
        self.cam.node().getLens().setAspectRatio(aspectRatio)
        self.inputmgr.check_mouse1drag()
        self.inputmgr.check_mouse2drag()
        self.inputmgr.check_mouse3click()
        self.inputmgr.check_mousewheel()
        self.inputmgr.check_resetcamera()
        return task.cont

    def __physicsUpdate(self, task):
        self.physicsworld.doPhysics(globalClock.getDt(), 20, 1.0/1200.0)
        return task.cont

    def __renderUpdate(self, task):
        for otdele in self.__objtodraw:
            otdele.detachNode()
            otdele.reparentTo(base.render)
        return task.cont

    def __rotatecamUpdate(self, task):
        campos = self.cam.getPos()
        camangle = math.atan2(campos[1], campos[0])
        # print camangle
        if camangle < 0:
            camangle += math.pi*2
        if camangle >= math.pi*2:
            camangle = 0
        else:
            camangle += math.pi/180
        camradius = math.sqrt(campos[0]*campos[0]+campos[1]*campos[1])
        camx = camradius*math.cos(camangle)
        camy= camradius*math.sin(camangle)
        self.cam.setPos(camx, camy, campos[2])
        self.cam.lookAt(self.lookatp[0], self.lookatp[1], self.lookatp[2])
        return task.cont

    def attachRUD(self, *args):
        """
        add to the render update list

        *args,**kwargs
        :param obj: nodepath, collision model, or bullet dynamics model
        :return:

        author: weiwei
        date: 20190627
        """

        for obj in args:
            self.__objtodraw.append(obj)

    def detachRUD(self, *args):
        """
        remove from the render update list

        :param obj: nodepath, collision model, or bullet dynamics model
        :return:

        author: weiwei
        date: 20190627
        """

        for obj in args:
            self.__objtodraw.remove(obj)

    def removeFromRUD(self, obj):
        """
        add to render with update

        :param obj: nodepath, collision model, or bullet dynamics model
        :return:

        author: weiwei
        date: 20190627
        """
        self.__objtodraw.append(obj)


    def changeLookAt(self, lookatp):
        """
        This function is questionable
        as lookat changes the rotation of the camera

        :param lookatp:
        :return:

        author: weiwei
        date: 20180606
        """

        self.cam.lookAt(lookatp[0], lookatp[1], lookatp[2])
        self.inputmgr = im.InputManager(self, lookatp, self.pggen)

    def setCartoonShader(self, switchtoon = False):
        """
        set cartoon shader, the following program is a reference
        https://github.com/panda3d/panda3d/blob/master/samples/cartoon-shader/advanced.py

        :return:

        author: weiwei
        date: 20180601
        """

        this_dir, this_filename = os.path.split(__file__)
        if switchtoon:
            lightinggen = Filename.fromOsSpecific(os.path.join(this_dir, "shaders", "lightingGen.sha"))
            tempnode = NodePath("temp")
            tempnode.setShader(loader.loadShader(lightinggen))
            self.cam.node().setInitialState(tempnode.getState())
            # self.render.setShaderInput("light", self.cam)
            self.render.setShaderInput("light", self.__ptlightnode0)
        #
        normalsBuffer = self.win.makeTextureBuffer("normalsBuffer", 0, 0)
        normalsBuffer.setClearColor(Vec4(0.5, 0.5, 0.5, 1))
        normalsCamera = self.makeCamera(
            normalsBuffer, lens=self.cam.node().getLens(), scene = self.render)
        normalsCamera.reparentTo(self.cam)
        normalgen = Filename.fromOsSpecific(os.path.join(this_dir, "shaders", "normalGen.sha"))
        tempnode = NodePath("temp")
        tempnode.setShader(loader.loadShader(normalgen))
        normalsCamera.node().setInitialState(tempnode.getState())

        drawnScene = normalsBuffer.getTextureCard()
        drawnScene.setTransparency(1)
        drawnScene.setColor(1, 1, 1, 0)
        drawnScene.reparentTo(render2d)
        self.drawnScene = drawnScene
        self.separation = 0.0007
        self.cutoff = 0.05
        normalgen = Filename.fromOsSpecific(os.path.join(this_dir, "shaders", "inkGen.sha"))
        drawnScene.setShader(loader.loadShader(normalgen))
        drawnScene.setShaderInput("separation", Vec4(self.separation, 0, self.separation, 0))
        drawnScene.setShaderInput("cutoff", Vec4(self.cutoff))