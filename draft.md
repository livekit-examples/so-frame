# making zero-shot sim2real possible on so-frame

for the past months, we've been designing a cheap robot rig for benchmarking our transport at livekit, called so-frame.

while our benchmarking focus so far has been on training behavior cloning models and collecting data for them, it has always been a core question of mine (after hours of collecting data): "can't the robot self-learn these behaviors?"

i'm so tired of collecting data while our focus is on infra, and, after all, we are doing very simple tasks such as picking and placing objects in a very controlled environment.

so as we released so-frame to the world, we also designed and released its digital twin, with the vision that we will train a rl model that reduces our need to collect data for the rig, be it rl from scratch or from prior demonstrations.

our aim is very simple: make sim2real work end to end, the same way our bc policies work, purely through visual and proprioceptive states.

today, we got it to work reliably and this is a write-up on how we did it.

<!-- IMG 1 (hero): matched sim | real rollout. see blog-viz/README.md -->

> _hero clip pending._

# what is the environment?

our robot is an SO-101 5-DOF arm mounted on a linear rail, giving 7 actuated DOF total:

- `dof_slider`, the rail (linear travel along the work surface)
- `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, the arm
- `gripper`, a parallel jaw

the arm and rail are bolted to a frame with a diffuse lightbox work surface (near-white, evenly lit). two cameras are rigidly mounted to the frame via printed holders:

- wrist camera, follows the gripper, used for fine grasp alignment
- overhead camera, static, sees the whole work surface, used to localize the objects

both are cheap camera modules, the specifications of which are not important to reproduction, as we built a tool that helps us align simulation cameras with real cameras, which will be mentioned later. the entire frame, arm, rail, cameras, and lightbox panels are one URDF, so the simulated twin and the real rig share the same kinematics and the same calibrated camera mounts (camera poses come from forward kinematics of the URDF links, not hand-measured offsets).

we call this rig the so-frame and its full description can be found [here](https://github.com/livekit-examples/so-frame).

# what is the task?

the task for the robot is to pick up a cube on the work surface and place it in a bin.

- cube: 20 mm, ~3.2 g, blue.
- bin: 100 mm square, 30 mm tall, 2 mm walls, so a 96 mm opening 28 mm deep. yellow.

in the simulation, the cube and bin's positions and rotations are randomized for each episode. both come from one zone, 358 × 728 mm, the bin placed first and the cube rejection-sampled until it clears the bin by 50 mm.

that zone is the overhead camera's footprint, measured at both the cube's height and the taller bin's rim, and inset on the far edge to keep it inside the arm's top-down reach. the policy is vision only, so a spawn out of frame would be unobservable rather than merely hard.

![the spawn zone, and episodes drawn from it](blog-viz/out/fig5_spawn_zone.png)

the task is considered successful if all of the following hold: the cube settles inside the bin, the cube and the robot are both static, and the robot touches neither the cube nor the bin. episodes are capped at 200 steps, each step one action. the robot is controlled at 10 hz, so each episode has 20 seconds.

the simulator we use is [ManiSkill3](https://arxiv.org/abs/2410.00425) ([repo](https://github.com/haosulab/ManiSkill)) from [Sapien](https://github.com/haosulab/SAPIEN). it is a popular framework with state of the art visual rendering, which is why i picked it, as we want to focus on visual learning instead of the state-based learning mostly used in locomotion. furthermore, ManiSkill's author has an [official implementation for sim2real on the so101](https://github.com/StoneT2000/lerobot-sim2real) which is of great reference.

# formalizing the problem

this part can be hard for some readers, as i'll formalize the RL environment mathematically as a Markov Decision Process. if you don't know what that is, i recommend watching [this video](https://www.youtube.com/watch?v=KZeIEiBrT_w), as well as reading up on [Spinning Up](https://spinningup.openai.com/en/latest/) from OpenAI.

I like formalizing as it's easier for me to explain, aka it has the highest information density.

If you have learned about MDP, our problem is a POMDP, a partially observable markov decision process. the agent never sees the true state (object poses, physics parameters), only what the sensors let through. This includes the two camera frames, plus 7 proprioceptive joint states and current commanded actions:

$$o_t = \Big(\phi(\mathbf{X}_t),\ \big[\tilde{\mathbf{q}}_t,\ \mathbf{q}^{\text{tgt}}_t\big]\Big)$$

$\phi$ is the encoder's preprocessing, and the choice of it is most of this article.

$\tilde{\mathbf{q}}_t$ is the 7 measured joint positions, $\mathcal{N}(0, \sigma_q^2)$ with $\sigma_q = 5°$ per joint to model real encoders.

$\mathbf{q}^{\text{tgt}}_t$ is the running controller target. the controller is positional, but command actions are not instantaneous, so $\tilde{\mathbf{q}}_t$ alone is ambiguous, so we need to model control delay as well, especially since our action space is delta.

that layout is a contract, not a convention. the 14 fields and their order are measured off the live training env and written into the checkpoint, so deploy assembles its vector from that record instead of assuming a width.

the action is a normalized delta, integrated into that target and clipped to the joint limits:

$$\mathbf{q}^{\text{tgt}}_{t+1} = \mathrm{clip}\!\big(\mathbf{q}^{\text{tgt}}_t + \mathbf{a}_t \odot \boldsymbol{\Delta}_{\max},\ \mathbf{q}_{\text{lo}},\ \mathbf{q}_{\text{hi}}\big), \qquad \mathbf{a}_t \in [-1,1]^7$$

so an action of $+1$ moves that joint's target by its full step. a PD controller chases the target at 100 hz while actions arrive at 10 hz. the gripper is continuous, all the way to the real robot.

note that a delta per control period is a velocity. that matters later.

### the reward

the reward is a staircase over five mutually exclusive stages. each stage has a fixed rung, plus at most a bounded amount of shaping on top, and the rungs are spaced so that a stage's maximum still sits below the next stage's rung, which means a higher stage always overrides the one below it instead of adding to it.

first the predicates, all read off the sim state:

- $\mathrm{G}$: the cube is grasped, both jaws in contact and closing on it
- $\mathrm{B}$: the cube is horizontally inside the bin's 96 mm opening, $|p^{x}_{\text{item}} - p^{x}_{\text{bin}}| < 0.048$ and likewise in $y$
- $\mathrm{T}$: the robot is touching the cube
- $\mathrm{I}$: the cube is _in_ the bin, $\mathrm{B}$ and its lowest corner within 5 mm of the bin floor
- $\Sigma = \mathrm{I} \wedge \|\dot p_{\text{item}}\| \le 0.02 \wedge \neg\mathrm{T} \wedge \text{robot static} \wedge \neg\text{touching the bin}$

and the distances: $d_{xy}$ and $d_z$ from the tool centre to the cube, $d_g = \|g - p_{\text{item}}\|$ from the cube to the drop point $g$, and $o \in [0,1]$ for how far the jaw is open.

| stage                    | condition                          | reward                                      |
| ------------------------ | ---------------------------------- | ------------------------------------------- |
| a. reach                 | otherwise                          | $r_{\text{reach}} \in [0, 1.5]$             |
| b. grasped               | $\mathrm{G} \wedge \neg\mathrm{B}$ | $2 + \big(1 - \tanh(5 d_g)\big) \in [2, 3]$ |
| c. holding over the bin  | $\mathrm{B} \wedge \mathrm{T}$     | $4 + o \in [4, 5]$                          |
| d. released over the bin | $\mathrm{B} \wedge \neg\mathrm{T}$ | $6$                                         |
| e. success               | $\Sigma$                           | $10$                                        |

$$r_{\text{reach}} = \underbrace{0.5\big(1 - \tanh(5\,d_{xy})\big)}_{\text{align over the cube}} \;+\; \mathbb{1}[\,d_{xy} < 0.03\,]\underbrace{0.5\big(1 - \tanh(5\,d_z)\big)}_{\text{then descend}} \;+\; \mathbb{1}[\,d_{xy} < 0.03 \,\wedge\, d_z < 0.02\,]\underbrace{0.5\,(1 - o)}_{\text{then close}}$$

the per-step reward is normalized by the maximum, $\hat r_t = r_t / 10$, and what the policy maximizes is the discounted sum over the episode, $\sum_t \gamma^t \hat r_t$ at $\gamma = 0.9$. so the ladder is a rate, not a score for finishing: more steps spent on a higher rung is worth more.

![the reward ladder](blog-viz/out/fig2_reward_ladder.png)

d and e are easy to conflate. d fires the instant the jaw stops touching a cube that is over the opening, and says nothing about where that cube ends up: it can still be in the air, it can catch the rim and bounce out, and the arm can be leaning on the bin throughout. e needs the outcome, cube down on the floor and slow, arm stopped, robot touching neither. d is the decision to let go, e is that decision having worked, which is why d is flat: nothing left to shape, the only way up is for the throw to land.

the ordering replaces a pile of bonuses and hover taxes. "let go, don't hover" is $5 < 6$, and regression handles itself, since dropping the cube falls to a lower rung with no penalty needed.

the two jaw terms are mirror images and took longest to get right. opening pays in c, so a jaw opening over the bin climbs toward d instead of leaping a plateau; closing pays at the top of a, but only with the tool on the cube in both axes. both are capped below the next rung, so a jaw shutting on nothing loses to a grasp, and the most open still-holding pose loses to a release. an earlier version made the gripper binary to force a clean release, which was treating a reward problem as an action-space problem.

no penalty terms anywhere: speed is capped by the action space and torque by the servos' stall.

the drop point $g$ sits 5 cm above the rim rather than on the floor. the opening is 96 mm and the jaw cannot swing open at depth inside it, so shaping toward a deep insert teaches the policy into being unable to let go. aiming high leaves the last few centimetres to gravity.

# matching sim2real

a policy trained inside one simulator ends up learning that simulator, its exact lighting and its exact frictions and its exact camera pose, so the moment i drop it onto the real rig all of those are slightly wrong at once and it falls apart.

there are two halves to closing that gap and i find it worth keeping them apart in my head. the first is alignment, where i make the simulator agree with the one real rig i actually own, so that the distribution the policy trains in is centred on the robot it will end up driving. the second is randomization, where i vary everything that changes from one run of that rig to the next, so the policy never comes to depend on any single draw of it.

## real alignment to sim

calibrating the cameras turned out to be the easy half, which honestly surprised me. because the camera holders are part of the so-frame URDF, the simulated camera already sits exactly where the real one is bolted and its pose falls out of forward kinematics, so i never had to measure an offset or fit a pose by hand. the only thing left standing between the two views is the lens.

that lens is a cheap 120° wide-angle module with real barrel distortion, while the simulator renders a clean pinhole, and the two end up seeing quite different fractions of the same scene. rather than teach the renderer to imitate a cheap lens i went the other way and rectified reality into the simulator's geometry, which is really just undistorting with a $k_1/k_2$ plus focal model, rotating (the overhead camera is mounted sideways), zooming and cropping down to the field of view sim renders, and then correcting the colour with a per-channel gain and a gamma.

<!-- IMG: screenshot of the calibration tool mid-fit. -->

> _tool screenshot pending._

i fit those parameters in a tool that drives the arm while rendering the sim cameras live beside the rectified real feed, with a blend slider between them so i can watch the two converge as i turn each knob, and it writes out the same mapping file the deploy loop later replays, which means the frame the policy trains on and the frame it sees on the robot are formed by an identical transform. driving the arm while fitting is the part i would not skip, because the wrist camera sees almost nothing except its own jaws, and a fit that looks perfect at one pose can be badly wrong at the next.

![rectifying reality into the simulator](blog-viz/out/fig3_calibration.png)

i treat that figure as a check rather than a claim, so sim is rendered at the exact joint pose the real frames were captured at, and the cube and bin are stood wherever the rectified real frame says they are, recovered by un-projecting them back onto the work surface. the arm and the objects and the frame extrusions all landing on top of each other is what tells me the mapping holds, and it also shows me the one loose end i have left: the overhead camera's field of view was fitted against the rig while the wrist's is still inherited from the MJCF twin, which is why its objects sit slightly too large.

**colours and lighting.** the linear base colours in sim are picked so they land on the real ones under sim's own lighting, and nothing in the scene is ever pure black or pure white, since a pure black object returns no light and renders as a flat silhouette with no shape left in it while a pure white one clips under the softbox and loses its edges the same way. the shadow-casting key light is switched off entirely, because the real lightbox throws no directional shadow at all, only a little contact darkening at the base of an object, and a cast shadow in sim is just one more artifact for the policy to key on.

**speed.** the STS3215 servos are slow, the rail especially, and a policy trained to command motion the hardware cannot actually track will wind its target up ahead of the arm until the arm overshoots and starts oscillating. so i measured what the real thing does, driving each joint to its limit and reading the achieved speed straight off the observation stream, which came out at 29 to 34 deg/s on the arm and about 7 cm/s on the rail, and sim's per-step deltas come from exactly those numbers, 0.05 rad and 7 mm per step at 10 hz. worth saying that speed is enforced by the action space rather than by torque, since 3 N·m against the servo's damping would reach something like 280 deg/s, an order of magnitude past anything the real arm does.

**torque.** 3 N·m is where those servos stall, so that is what each joint's force limit is set to in sim. an over-powered sim arm learns to muscle its way through an imprecise grasp and to lean on the work surface, and neither of those transfers, because the real servo simply stalls instead. at the real stall torque the low top-down grasp actually gets easier, since a weak arm settles onto the cube rather than slamming into it and bouncing off.

## domain randomize

alignment centres the distribution and randomization is what gives it width, so that reality reads as one more sample out of it rather than a point sitting outside it.

![domain randomization draws](blog-viz/out/fig6_domain_randomization.png)

| randomization       | range                                                                  |
| ------------------- | ---------------------------------------------------------------------- |
| ambient lighting    | 0.2 to 0.5 per channel                                                 |
| camera pose and FOV | ±2 mm, ±1°, ±1°                                                        |
| gripper gains       | stiffness 500 to 2000, damping 50 to 200                               |
| arm and rail gains  | stiffness 600 to 1400, damping 60 to 140                               |
| proprio noise       | 5° std on joint reads                                                  |
| cube friction       | 0.5 to 1.0                                                             |
| colour jitter       | brightness/contrast/saturation 0.3, hue 0.05, per camera               |
| sensor realism      | gamma 0.7 to 1.4, ±10% white balance, noise, blur, a compression proxy |

camera pose and FOV are drawn once when the scene is built rather than every episode, since those are properties of a rig and not of a moment.

the one thing i deliberately leave alone is the colour of the task objects. randomizing it costs real sample efficiency, and there is exactly one physical rig whose cube and bin are a known blue and a known yellow, so i match them instead and let the policy treat colour as a cue it can rely on. how heavily a given encoder ends up leaning on that cue turns out to matter enormously, which is where the rest of this article goes.

# improving upon prior work

with our environment defined mathematically, we'll need to worry about how do we train our policy: this means we'll have to pick our rl method, would it be value based, or policy gradient, or would it be both?

### 1. squint

this leads us to [squint](https://arxiv.org/abs/2602.21203), which is a zero shot sim2real method on the SO101 published in february this year.

thanks to [pratham](https://x.com/PrathamJainAI/status/2076232338447724623) for introducing me to this method on his twitter thread and sharing his squint notes with me.

the answer to the question above is both. squint is a visual [Soft Actor-Critic](https://arxiv.org/abs/1801.01290), off-policy and entropy-regularized, so parallel simulation can fill a replay buffer fast while a learned critic squeezes many gradient updates out of every environment step. instead of the plain discounted return it maximizes return plus an entropy bonus,

$$\pi^\star = \arg\max_\pi\ \mathbb{E}_\pi\!\Big[\textstyle\sum_t \gamma^t\big(r_t + \alpha\,\mathcal{H}(\pi(\cdot\mid o_t))\big)\Big], \qquad \gamma = 0.9$$

which pays the policy to stay stochastic rather than collapse onto the first behavior that scores. the temperature $\alpha$ is auto-tuned so exploration fades on its own, and at deploy the actor is deterministic.

there are three networks. a small conv stack over the two cameras stacked to H×W×6. an MLP actor that fuses the visual features with proprio. and a [distributional critic](https://arxiv.org/abs/1707.06887): instead of a scalar Q-value each critic predicts a distribution over returns across 101 atoms spanning $[-20, 20]$, evenly spaced candidate returns with a probability on each. the Bellman target shifts every atom by $r + \gamma z_i$, projects it back onto the fixed grid, and the loss is a cross-entropy over a two-network ensemble. for a staged reward like ours this is much more stable than scalar Q-learning.

then the trick it is named for. the cameras render at 128×128 and are area-downsampled to 32×32 before the network sees them. the policy squints. this beats rendering natively at 32 because a native 32 px render point-samples the scene, so a small object flickers or vanishes between frames, while averaging a 4×4 block leaves a stable soft signal. and it is fast, a full run in well under two hours on one GPU.

but look at what squinting costs on this task.

![what each encoder is handed](blog-viz/out/fig4_what_the_encoder_sees.png)

the cube is 20 mm on a 710 mm workspace. it covers a 5×5 block of the 128 px render, and **two pixels** of the 32 px squint. no shape, no edges, no orientation. the only property that survives is hue, which is why every run here paints the cube blue and the bin yellow instead of the black they are by default. at 32 px, colour is the only cue the CNN has.

hold that thought.

### 2. dinov2 encoder

squint has a problem when going to higher resolution, which is that it focuses too much on unwanted visual artifacts such as shadows and wires. a CNN trained purely inside a simulator has never seen the real world, so it has no prior for what is signal and what is a rendering artifact. to improve upon this, i then switched the from-scratch CNN encoder for a [DINOv2](https://arxiv.org/abs/2304.07193) pre-trained encoder, ViT-S/14 [with registers](https://arxiv.org/abs/2309.16588), trained self-supervised on real images at scale.

it stays frozen on purpose. fine-tuning it on sim renders would just re-teach it the simulator's quirks and throw away the prior that motivated the swap.

![dinov2 features, overhead camera](blog-viz/out/fig7_dino_features.png)

the middle column is the reason to bother. the same frozen backbone on a sim render and on a rectified real frame, both painted by a single shared PCA so the colours are comparable rather than each image being flattered by its own projection. the surface, the arm and the frame edges take the same colours in both worlds, and we did not have to train for it.

the mistake to avoid is treating DINOv2 like a CNN, flattening its output into one vector and moving on. it does not hand you a feature image, it hands you tokens. at 168 px, twelve of its 14-pixel patches a side, each camera is a 12×12 grid of 384-dim patch tokens, 288 for the pair. so the head consumes tokens: both grids go in jointly with a learned per-camera embedding, self-attention runs over the sequence, and a learned readout token collects the answer. this follows the [Patch Policy](https://arxiv.org/abs/2607.18236) recipe, whose claim is exactly that dense representations are what embodied control needs.

![dinov2 features, wrist camera](blog-viz/out/fig8_dino_features_wrist.png)

the wrist view is worth its own figure because it is a different kind of evidence. the overhead camera shows the backbone agreeing about a scene, while the wrist camera sees almost nothing except its own gripper, so it shows the backbone agreeing about the tool, which is what the grasp actually depends on.

that is testable rather than believable, so i built the control. `dino_global` is the same frozen backbone at the same resolution with the same head width and update ratio, differing in one thing: the patch grid is collapsed to one vector per camera before the head sees it, either the CLS token or the mean over patches. 288 tokens against 2.

one practical note. since the backbone is frozen its tokens never change for a given frame, so they are computed once per environment step in an observation wrapper and cached in the replay buffer instead of recomputed on every gradient batch.

# training notes

training is 12M environment steps on a single RTX PRO 6000, replay retention of 2 episodes per env, batch 512, and squint's hyperparameters mostly untouched. the squint CNN runs 1024 parallel envs at 2833 environment steps per second and finishes in about ninety minutes. the dino heads run 512 envs, and the patch head is the expensive one at 341 steps per second and ten hours.

four encoders, same task, same reward, same retention, same budget.

![evaluation success by encoder](blog-viz/out/fig1_encoder_curves.png)

| encoder                    | first success | best     | sustained |
| -------------------------- | ------------- | -------- | --------- |
| `dino_patch`, 12×12 grid   | 1.75M         | **1.00** | **0.88**  |
| `dino_global`, mean-pooled | 2.75M         | 0.89     | 0.67      |
| `dino_global`, CLS token   | 7.75M         | 0.80     | 0.58      |
| `squint` CNN at 32 px      | 3.50M         | 0.71     | 0.52      |

the dense grid wins on both axes. it blooms first and holds the highest level once it is there. collapsing the same features to one vector per camera costs about twenty points of sustained success, and collapsing to the CLS token costs another nine and delays the bloom by five million steps. the only difference between the top row and the middle two is whether the patch grid survives to the head.

the right panel of that figure is the part i would most want you to take away. before the reward ladder gained its jaw-closing ramp and the horizon came down to 200 steps, the same CNN ran five times, three of them separate seeds, twelve million steps each. it never placed the cube once. not rarely, never. nothing about the encoder changed between those runs and the one that reached 0.71. i spent a while suspecting the architecture when the reward was the problem.

three more things from the run history:

- **entropy collapse is normal.** the auto-tuned temperature bottoms out around 1e-4 in the first 600k steps of every run, successful or not. printed with three decimals it reads as 0.000, which makes it look worse than it is.
- **the final eval is not the number you want.** the patch head does not converge, it wanders. after peaking at 1.00 it spent seven million more steps between 0.69 and 1.00, ending at 0.83. an earlier run of the same architecture did the ugly version, peaking at 0.74 then falling to flat zero. every checkpoint deployed here is a best checkpoint, and the one driving the robot is from step 6.0M of a 12M run.
- **one run is not an experiment.** the CNN has high seed variance. one seed of a proven config finished at exactly 0.000 while another reached 0.688. eval is 35 episodes, so 0 out of 35 is not sampling noise off a working policy, it is a real policy state.

# deployment

my so-frame is built to be a remote rig and so i always connect to it remotely through [livekit portal](https://github.com/livekit/portal).

in this setup, the robot acts as a participant connecting to the policy (also as a participant) through a livekit room.

the robot exposes its controls while sending raw camera frames to the policy. on the policy side, a simple bridge is written to rectify reality to simulation constraints. sending raw frames is deliberate: the human watching the web UI gets the full wide-angle view, and the policy privately reconstructs its narrow 168 px version before every inference.

```mermaid
flowchart TD
    subgraph robot["robot runtime"]
        cams["two raw frames<br/>640×480, 120° DFOV"]
        qpos["measured joint state"]
        servos["servos track the target"]
    end

    subgraph policy["policy operator"]
        rect["rectify per camera<br/>rotate, undistort, zoom, crop, colour"]
        stack["stack wrist + overhead<br/>168×168×6"]
        tok["frozen DINOv2<br/>288 patch tokens"]
        actor["actor mean<br/>delta action in [-1,1]^7"]
        integrate["integrate running target"]
        bridge["sim units to wire units"]
    end

    cams -- livekit room --> rect
    qpos -- livekit room --> actor
    rect --> stack --> tok --> actor --> integrate --> bridge
    bridge -- joint targets --> servos
    integrate -. next tick's state .-> actor
```

the bridge itself is thin, so the network's tensors never leave sim space: radians to degrees for the arm, metres to a normalized 0-to-100 position for the rail. one joint, `wrist_roll`, carries a measured 90° offset because its calibrated zero is not the URDF zero. everything else is identity, checked against a live arm rather than assumed.

back to actions being velocities. sim integrates one every step, so deploy keeps applying an action every tick until a new one replaces it.

the real arm lags its target where the sim arm does not, so a target left to advance freely winds up, and the jaw closes after the arm has already gone past the cube. the fix is a lag budget: the target only advances while every gated joint is within a few action steps of its measured pose, and the same threshold gates the next decision. it trades directly against speed.

| budget | decisions/s | rail speed | 0.49 m traverse |
| ------ | ----------- | ---------- | --------------- |
| 0.5    | 1.4         | 1.00 cm/s  | 49.4 s          |
| 1.0    | 2.3         | 1.55 cm/s  | 31.8 s          |
| 2.0    | 3.8         | 2.60 cm/s  | 19.0 s          |
| 4.0    | 6.8         | 4.53 cm/s  | 10.9 s          |

two exclusions earn their place. the gripper is exempt from the shared budget, because its whole range is 9.6 action steps and a jaw closed on the cube sits several steps short of its command for as long as it holds, so a shared gate would never clear again. it carries its own generous lead cap instead, since that lead is the grip force: a position servo only pushes as hard as the distance it is asked to close. and nothing advances without frames, because a lost camera must not mean the arm keeps gliding blind on a stale command.

> _deploy `--viz` screenshot pending._

# takeaway

this is just a simple proof of concept (which is reliable and reproducible), it took me 1 week for the rl env and another to successfully adapt squint and more to the rl environment and to real life.

some learnings as of now:

- there is nothing i can do in sim to reliably compensate for how crap the sts3215 response is. it just sucks. i can throw a bunch of domain randomization at it, but for the love of god, i don't want to do it like that.
- pretrained models help generalization and improve sample efficiency (u can say duh sherlock). in my case, only my dino backbone models transfer to the real world, so it holds some weight, or i can say damn, maybe i'm stupid with doing squint.
- patch policy greatly improves policy performance. global mean and cls token cannot compare, in sim or in real. i spent a good amount of time trying to figure out a way there, then [the paper dropped out of nowhere](https://x.com/jeffacce/status/2080017577749684718?s=20) and i was like this is it. god level timing.
- visual is one thing, matching control behaviors and system delay is a whole other problem set, which i have not solved.
