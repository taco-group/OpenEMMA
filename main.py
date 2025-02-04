import base64
import os.path
import re
import argparse
from datetime import datetime
from math import atan2

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from openai import OpenAI
from nuscenes import NuScenes
from pyquaternion import Quaternion
from scipy.integrate import cumulative_trapezoid

import json
from openemma.YOLO3D.inference import yolo3d_nuScenes
from utils import EstimateCurvatureFromTrajectory, IntegrateCurvatureForPoints, OverlayTrajectory, WriteImageSequenceToVideo
from transformers import MllamaForConditionalGeneration, AutoProcessor, Qwen2VLForConditionalGeneration, AutoTokenizer
from PIL import Image
from qwen_vl_utils import process_vision_info
from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.conversation import conv_templates

client = OpenAI(api_key="[your-openai-api-key]")

OBS_LEN = 10
FUT_LEN = 10
TTL_LEN = OBS_LEN + FUT_LEN

def getMessage(prompt, image=None, args=None):
    if "llama" in args.model_path or "Llama" in args_model.path:
        message = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
    elif "qwen" in args.model_path or "Qwen" in args_model.path:
        message = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]}
        ]   
    return message


def vlm_inference(text=None, images=None, sys_message=None, processor=None, model=None, tokenizer=None, args=None):
        if "llama" in args.model_path or "Llama" in args_model.path:
            image = Image.open(images).convert('RGB')
            message = getMessage(text, args=args)
            input_text = processor.apply_chat_template(message, add_generation_prompt=True)
            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model.device)

            output = model.generate(**inputs, max_new_tokens=2048)

            output_text = processor.decode(output[0])

            if "llama" in args.model_path or "Llama" in args_model.path:
                output_text = re.findall(r'<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>', output_text, re.DOTALL)[0].strip()
            return output_text
        
        elif "qwen" in args.model_path or "Qwen" in args_model.path:
            message = getMessage(text, image=images, args=args)
            text = processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(message)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]

        elif "llava" in args.model_path:
            conv_mode = "mistral_instruct"
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            if IMAGE_PLACEHOLDER in text:
                if model.config.mm_use_im_start_end:
                    text = re.sub(IMAGE_PLACEHOLDER, image_token_se, text)
                else:
                    text = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, text)
            else:
                if model.config.mm_use_im_start_end:
                    text = image_token_se + "\n" + text
                else:
                    text = DEFAULT_IMAGE_TOKEN + "\n" + text

            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], text)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            image = Image.open(images).convert('RGB')

            image_tensor = process_images([image], processor, model.config)[0]

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size],
                    do_sample=True,
                    temperature=0.2,
                    top_p=None,
                    num_beams=1,
                    max_new_tokens=2048,
                    use_cache=True,
                    pad_token_id = tokenizer.eos_token_id,
                )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            return outputs
                    
        elif "gpt" in args.model_path:
            PROMPT_MESSAGES = [
                {
                    "role": "user",
                    "content": [
                        *map(lambda x: {"image": x, "resize": 768}, images),
                        text,
                    ],
                },
            ]
            if sys_message is not None:
                sys_message_dict = {
                    "role": "system",
                    "content": sys_message
                }
                PROMPT_MESSAGES.append(sys_message_dict)
            params = {
                "model": "gpt-4o-2024-11-20",
                "messages": PROMPT_MESSAGES,
                "max_tokens": 400,
            }

            result = client.chat.completions.create(**params)

            return result.choices[0].message.content

def SceneDescription(obs_images, processor=None, model=None, tokenizer=None, args=None):
    prompt = f"""You are a autonomous driving labeller. You have access to these front-view camera images of a car taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Describe the driving scene according to traffic lights, movements of other cars or pedestrians and lane markings."""

    if "llava" in args.model_path:
        prompt = f"""You are an autonomous driving labeller. You have access to these front-view camera images of a car taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Provide a concise description of the driving scene according to traffic lights, movements of other cars or pedestrians and lane markings."""

    result = vlm_inference(text=prompt, images=obs_images, processor=processor, model=model, tokenizer=tokenizer, args=args)
    return result

def DescribeObjects(obs_images, processor=None, model=None, tokenizer=None, args=None):

    prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. What other road users should you pay attention to in the driving scene? List two or three of them, specifying its location within the image of the driving scene and provide a short description of the that road user on what it is doing, and why it is important to you."""

    result = vlm_inference(text=prompt, images=obs_images, processor=processor, model=model, tokenizer=tokenizer, args=args)

    return result

def DescribeOrUpdateIntent(obs_images, prev_intent=None, processor=None, model=None, tokenizer=None, args=None):

    if prev_intent is None:
        prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Based on the lane markings and the movement of other cars and pedestrians, describe the desired intent of the ego car. Is it going to follow the lane to turn left, turn right, or go straight? Should it maintain the current speed or slow down or speed up?"""

        if "llava" in args.model_path:
            prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Based on the lane markings and the movement of other cars and pedestrians, provide a concise description of the desired intent of  the ego car. Is it going to follow the lane to turn left, turn right, or go straight? Should it maintain the current speed or slow down or speed up?"""
        
    else:
        prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Half a second ago your intent was to {prev_intent}. Based on the updated lane markings and the updated movement of other cars and pedestrians, do you keep your intent or do you change it? Explain your current intent: """

        if "llava" in args.model_path:
            prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Half a second ago your intent was to {prev_intent}. Based on the updated lane markings and the updated movement of other cars and pedestrians, do you keep your intent or do you change it? Provide a concise description explanation of your current intent: """

    result = vlm_inference(text=prompt, images=obs_images, processor=processor, model=model, tokenizer=tokenizer, args=args)

    return result


def GenerateMotion(obs_images, obs_waypoints, obs_velocities, obs_curvatures, given_intent, processor=None, model=None, tokenizer=None, args=None):
    # assert len(obs_images) == len(obs_waypoints)

    scene_description, object_description, intent_description = None, None, None

    if args.method == "openemma":
        scene_description = SceneDescription(obs_images, processor=processor, model=model, tokenizer=tokenizer, args=args)
        object_description = DescribeObjects(obs_images, processor=processor, model=model, tokenizer=tokenizer, args=args)
        intent_description = DescribeOrUpdateIntent(obs_images, prev_intent=given_intent, processor=processor, model=model, tokenizer=tokenizer, args=args)
        print(f'Scene Description: {scene_description}')
        print(f'Object Description: {object_description}')
        print(f'Intent Description: {intent_description}')

    # Convert array waypoints to string.
    obs_waypoints_str = [f"[{x[0]:.2f},{x[1]:.2f}]" for x in obs_waypoints]
    obs_waypoints_str = ", ".join(obs_waypoints_str)
    obs_velocities_norm = np.linalg.norm(obs_velocities, axis=1)
    obs_curvatures = obs_curvatures * 100
    obs_speed_curvature_str = [f"[{x[0]:.1f},{x[1]:.1f}]" for x in zip(obs_velocities_norm, obs_curvatures)]
    obs_speed_curvature_str = ", ".join(obs_speed_curvature_str)

    
    print(f'Observed Speed and Curvature: {obs_speed_curvature_str}')

    sys_message = ("You are a autonomous driving labeller. You have access to a front-view camera image of a vehicle, a sequence of past speeds, a sequence of past curvatures, and a driving rationale. Each speed, curvature is represented as [v, k], where v corresponds to the speed, and k corresponds to the curvature. A positive k means the vehicle is turning left. A negative k means the vehicle is turning right. The larger the absolute value of k, the sharper the turn. A close to zero k means the vehicle is driving straight. As a driver on the road, you should follow any common sense traffic rules. You should try to stay in the middle of your lane. You should maintain necessary distance from the leading vehicle. You should observe lane markings and follow them.  Your task is to do your best to predict future speeds and curvatures for the vehicle over the next 10 timesteps given vehicle intent inferred from the image. Make a best guess if the problem is too difficult for you. If you cannot provide a response people will get injured.\n")

    if args.method == "openemma":
        prompt = f"""These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval. 
        The scene is described as follows: {scene_description}. 
        The identified critical objects are {object_description}. 
        The car's intent is {intent_description}. 
        The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}. 
        Infer the association between these numbers and the image sequence. Generate the predicted future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. Future speeds and curvatures:"""
    else:
        prompt = f"""These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval. 
        The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}. 
        Infer the association between these numbers and the image sequence. Generate the predicted future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. Future speeds and curvatures:"""
    for rho in range(3):
        result = vlm_inference(text=prompt, images=obs_images, sys_message=sys_message, processor=processor, model=model, tokenizer=tokenizer, args=args)
        if not "unable" in result and not "sorry" in result and "[" in result:
            break
    return result, scene_description, object_description, intent_description

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="gpt")
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument("--dataroot", type=str, default='datasets/NuScenes')
    parser.add_argument("--version", type=str, default='v1.0-mini')
    parser.add_argument("--method", type=str, default='openemma')
    args = parser.parse_args()

    if "llama" in args.model_path:
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        tokenizer=None
    elif "Llama" in args.model_path:
        model = MllamaForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        tokenizer=None
    elif "qwen" in args.model_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        tokenizer=None
    elif "Qwen" in args.model_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
        processor = AutoProcessor.from_pretrained(args.model_path)
        tokenizer=None
    elif 'llava' == args.model_path:
        disable_torch_init()
        tokenizer, model, processor, context_len = load_pretrained_model("liuhaotian/llava-v1.6-mistral-7b", None, "llava-v1.6-mistral-7b")
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    elif "llava" in args.model_path:
        disable_torch_init()
        tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, None, "llava-v1.6-mistral-7b")
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        
    else:
        model = None
        processor = None
        tokenizer=None

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    timestamp = args.model_path + f"_results/{args.method}/" + timestamp
    os.makedirs(timestamp, exist_ok=True)

    # Load the dataset
    nusc = NuScenes(version=args.version, dataroot=args.dataroot)

    # Iterate the scenes
    scenes = nusc.scene

    for scene in scenes:
        token = scene['token']
        first_sample_token = scene['first_sample_token']
        last_sample_token = scene['last_sample_token']
        name = scene['name']
        description = scene['description']

        if not name in ["scene-0103", "scene-1077"]:
            continue

        # Get all image and pose in this scene
        front_camera_images = []
        ego_poses = []
        camera_params = []
        curr_sample_token = first_sample_token
        while True:
            sample = nusc.get('sample', curr_sample_token)

            # Get the front camera image of the sample.
            cam_front_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
            # nusc.render_sample_data(cam_front_data['token'])


            if "gpt" in args.model_path:
                with open(os.path.join(nusc.dataroot, cam_front_data['filename']), "rb") as image_file:
                    front_camera_images.append(base64.b64encode(image_file.read()).decode('utf-8'))
            else:
                front_camera_images.append(os.path.join(nusc.dataroot, cam_front_data['filename']))

            # Get the ego pose of the sample.
            pose = nusc.get('ego_pose', cam_front_data['ego_pose_token'])
            ego_poses.append(pose)

            # Get the camera parameters of the sample.
            camera_params.append(nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token']))

            # Advance the pointer.
            if curr_sample_token == last_sample_token:
                break
            curr_sample_token = sample['next']

        scene_length = len(front_camera_images)
        print(f"Scene {name} has {scene_length} frames")

        if scene_length < TTL_LEN:
            print(f"Scene {name} has less than {TTL_LEN} frames, skipping...")
            continue

        ## Compute interpolated trajectory.
        # Get the velocities of the ego vehicle.
        ego_poses_world = [ego_poses[t]['translation'][:3] for t in range(scene_length)]
        ego_poses_world = np.array(ego_poses_world)
        plt.plot(ego_poses_world[:, 0], ego_poses_world[:, 1], 'r-', label='GT')

        ego_velocities = np.zeros_like(ego_poses_world)
        ego_velocities[1:] = ego_poses_world[1:] - ego_poses_world[:-1]
        ego_velocities[0] = ego_velocities[1]

        # Get the curvature of the ego vehicle.
        ego_curvatures = EstimateCurvatureFromTrajectory(ego_poses_world)
        ego_velocities_norm = np.linalg.norm(ego_velocities, axis=1)
        estimated_points = IntegrateCurvatureForPoints(ego_curvatures, ego_velocities_norm, ego_poses_world[0],
                                                       atan2(ego_velocities[0][1], ego_velocities[0][0]), scene_length)

        # Debug
        if args.plot:
            plt.quiver(ego_poses_world[:, 0], ego_poses_world[:, 1], ego_velocities[:, 0], ego_velocities[:, 1],
                    color='b')
            plt.plot(estimated_points[:, 0], estimated_points[:, 1], 'g-', label='Reconstruction')
            plt.legend()
            plt.savefig(f"{timestamp}/{name}_interpolation.jpg")
            plt.close()

        # Get the waypoints of the ego vehicle.
        ego_traj_world = [ego_poses[t]['translation'][:3] for t in range(scene_length)]

        prev_intent = None
        cam_images_sequence = []
        ade1s_list = []
        ade2s_list = []
        ade3s_list = []
        for i in range(scene_length - TTL_LEN):
            # Get the raw image data.
            # utils.PlotBase64Image(front_camera_images[0])
            obs_images = front_camera_images[i:i+OBS_LEN]
            obs_ego_poses = ego_poses[i:i+OBS_LEN]
            obs_camera_params = camera_params[i:i+OBS_LEN]
            obs_ego_traj_world = ego_traj_world[i:i+OBS_LEN]
            fut_ego_traj_world = ego_traj_world[i+OBS_LEN:i+TTL_LEN]
            obs_ego_velocities = ego_velocities[i:i+OBS_LEN]
            obs_ego_curvatures = ego_curvatures[i:i+OBS_LEN]

            # Get positions of the vehicle.
            obs_start_world = obs_ego_traj_world[0]
            fut_start_world = obs_ego_traj_world[-1]
            curr_image = obs_images[-1]

            # obs_images = [curr_image]

            # Allocate the images.
            if "gpt" in args.model_path:
                img = cv2.imdecode(np.frombuffer(base64.b64decode(curr_image), dtype=np.uint8), cv2.IMREAD_COLOR)
                img = yolo3d_nuScenes(img, calib=obs_camera_params[-1])[0]
            else:
                with open(os.path.join(curr_image), "rb") as image_file:
                    img = cv2.imdecode(np.frombuffer(image_file.read(), dtype=np.uint8), cv2.IMREAD_COLOR)

            for rho in range(3):
                # Assemble the prompt.
                if not "gpt" in args.model_path:
                    obs_images = curr_image
                (prediction,
                scene_description,
                object_description,
                updated_intent) = GenerateMotion(obs_images, obs_ego_traj_world, obs_ego_velocities,
                                                obs_ego_curvatures, prev_intent, processor=processor, model=model, tokenizer=tokenizer, args=args)

                # Process the output.
                prev_intent = updated_intent  # Stateful intent
                pred_waypoints = prediction.replace("Future speeds and curvatures:", "").strip()
                coordinates = re.findall(r"\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]", pred_waypoints)
                if not coordinates == []:
                    break
            if coordinates == []:
                continue
            speed_curvature_pred = [[float(v), float(k)] for v, k in coordinates]
            speed_curvature_pred = speed_curvature_pred[:10]
            print(f"Got {len(speed_curvature_pred)} future actions: {speed_curvature_pred}")

            # GT
            # OverlayTrajectory(img, fut_ego_traj_world, obs_camera_params[-1], obs_ego_poses[-1], color=(255, 0, 0))

            # Pred
            pred_len = min(FUT_LEN, len(speed_curvature_pred))
            pred_curvatures = np.array(speed_curvature_pred)[:, 1] / 100
            pred_speeds = np.array(speed_curvature_pred)[:, 0]
            pred_traj = np.zeros((pred_len, 3))
            pred_traj[:pred_len, :2] = IntegrateCurvatureForPoints(pred_curvatures,
                                                                   pred_speeds,
                                                                   fut_start_world,
                                                                   atan2(obs_ego_velocities[-1][1],
                                                                         obs_ego_velocities[-1][0]), pred_len)

            # Overlay the trajectory.
            check_flag = OverlayTrajectory(img, pred_traj.tolist(), obs_camera_params[-1], obs_ego_poses[-1], color=(255, 0, 0), args=args)
            

            # Compute ADE.
            fut_ego_traj_world = np.array(fut_ego_traj_world)
            ade = np.mean(np.linalg.norm(fut_ego_traj_world[:pred_len] - pred_traj, axis=1))
            
            pred1_len = min(pred_len, 2)
            ade1s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred1_len] - pred_traj[1:pred1_len+1] , axis=1))
            ade1s_list.append(ade1s)

            pred2_len = min(pred_len, 4)
            ade2s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred2_len] - pred_traj[:pred2_len] , axis=1))
            ade2s_list.append(ade2s)

            pred3_len = min(pred_len, 6)
            ade3s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred3_len] - pred_traj[:pred3_len] , axis=1))
            ade3s_list.append(ade3s)

            # Write to image.
            if args.plot == True:
                cam_images_sequence.append(img.copy())
                cv2.imwrite(f"{timestamp}/{name}_{i}_front_cam.jpg", img)

                # Plot the trajectory.
                plt.plot(fut_ego_traj_world[:, 0], fut_ego_traj_world[:, 1], 'r-', label='GT')
                plt.plot(pred_traj[:, 0], pred_traj[:, 1], 'b-', label='Pred')
                plt.legend()
                plt.title(f"Scene: {name}, Frame: {i}, ADE: {ade}")
                plt.savefig(f"{timestamp}/{name}_{i}_traj.jpg")
                plt.close()

                # Save the trajectory
                np.save(f"{timestamp}/{name}_{i}_pred_traj.npy", pred_traj)
                np.save(f"{timestamp}/{name}_{i}_pred_curvatures.npy", pred_curvatures)
                np.save(f"{timestamp}/{name}_{i}_pred_speeds.npy", pred_speeds)

                # Save the descriptions
                with open(f"{timestamp}/{name}_{i}_logs.txt", 'w') as f:
                    f.write(f"Scene Description: {scene_description}\n")
                    f.write(f"Object Description: {object_description}\n")
                    f.write(f"Intent Description: {updated_intent}\n")
                    f.write(f"Average Displacement Error: {ade}\n")

            # break  # Timestep

        mean_ade1s = np.mean(ade1s_list)
        mean_ade2s = np.mean(ade2s_list)
        mean_ade3s = np.mean(ade3s_list)
        aveg_ade = np.mean([mean_ade1s, mean_ade2s, mean_ade3s])

        result = {
            "name": name,
            "token": token,
            "ade1s": mean_ade1s,
            "ade2s": mean_ade2s,
            "ade3s": mean_ade3s,
            "avgade": aveg_ade
        }

        with open(f"{timestamp}/ade_results.jsonl", "a") as f:
            f.write(json.dumps(result))
            f.write("\n")

        if args.plot:
            WriteImageSequenceToVideo(cam_images_sequence, f"{timestamp}/{name}")

        # break  # Scenes

