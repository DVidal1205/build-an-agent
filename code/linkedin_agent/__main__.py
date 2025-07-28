"""
Interactive CLI for LinkedIn Slop Bot with style questionnaire.
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent.parent))

from linkedin_agent import async_create_linkedin_post

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

def collect_style_preferences():
    """Collect style preferences from user on 1-5 scales."""
    
    print("\n🎨 STYLE QUESTIONNAIRE")
    print("=" * 50)
    print("Rate each preference on a scale of 1-5:")
    print("-" * 50)
    
    # Grammar level
    print("\n📝 GRAMMAR LEVEL:")
    print("1 = Terrible (typos, informal, text-speak)")
    print("2 = Poor (some mistakes, very casual)")
    print("3 = Casual (mostly correct, informal tone)")
    print("4 = Good (proper grammar, professional)")
    print("5 = Proper (perfect grammar, formal)")
    
    while True:
        try:
            grammar = int(input("Grammar level (1-5): "))
            if 1 <= grammar <= 5:
                break
            print("Please enter a number between 1 and 5")
        except ValueError:
            print("Please enter a valid number")
    
    # Emoji level
    print("\n😊 EMOJI LEVEL:")
    print("1 = Minimal (0-2 emojis)")
    print("2 = Few (2-4 emojis)")
    print("3 = Moderate (4-6 emojis)")
    print("4 = Many (8-12 emojis)")
    print("5 = Maximum (12+ emojis everywhere)")
    
    while True:
        try:
            emojis = int(input("Emoji level (1-5): "))
            if 1 <= emojis <= 5:
                break
            print("Please enter a number between 1 and 5")
        except ValueError:
            print("Please enter a valid number")
    
    # Hashtag level
    print("\n#️⃣ HASHTAG LEVEL:")
    print("1 = Few (3-5 hashtags)")
    print("2 = Some (5-7 hashtags)")
    print("3 = Moderate (6-10 hashtags)")
    print("4 = Many (10-15 hashtags)")
    print("5 = Maximum (15-20 hashtags)")
    
    while True:
        try:
            hashtags = int(input("Hashtag level (1-5): "))
            if 1 <= hashtags <= 5:
                break
            print("Please enter a number between 1 and 5")
        except ValueError:
            print("Please enter a valid number")
    
    # Ragebait level
    print("\n🔥 RAGEBAIT LEVEL (Egotistical/Self-absorbed):")
    print("1 = Humble (focus on others, modest)")
    print("2 = Modest (some self-mention, grateful)")
    print("3 = Balanced (normal self-promotion)")
    print("4 = Confident (strong self-promotion)")
    print("5 = Egotistical (highly self-absorbed, controversial)")
    
    while True:
        try:
            ragebait = int(input("Ragebait level (1-5): "))
            if 1 <= ragebait <= 5:
                break
            print("Please enter a number between 1 and 5")
        except ValueError:
            print("Please enter a valid number")
    
    # Inspirational level
    print("\n✨ INSPIRATIONAL LEVEL:")
    print("1 = Realistic (practical, down-to-earth)")
    print("2 = Practical (some motivation, mostly facts)")
    print("3 = Balanced (mix of inspiration and reality)")
    print("4 = Motivational (uplifting, encouraging)")
    print("5 = Dream-big (highly inspirational, aspirational)")
    
    while True:
        try:
            inspirational = int(input("Inspirational level (1-5): "))
            if 1 <= inspirational <= 5:
                break
            print("Please enter a number between 1 and 5")
        except ValueError:
            print("Please enter a valid number")
    
    # Informational level
    print("\n📊 INFORMATIONAL LEVEL:")
    print("1 = Opinion-based (personal views, stories)")
    print("2 = Story-focused (anecdotes, experiences)")
    print("3 = Balanced (mix of info and opinion)")
    print("4 = Educational (facts, insights, data)")
    print("5 = Data-rich (heavy on stats and industry info)")
    
    while True:
        try:
            informational = int(input("Informational level (1-5): "))
            if 1 <= informational <= 5:
                break
            print("Please enter a number between 1 and 5")
        except ValueError:
            print("Please enter a valid number")
    
    return {
        "grammar_level": grammar,
        "emoji_level": emojis,
        "hashtag_level": hashtags,
        "ragebait_level": ragebait,
        "inspirational_level": inspirational,
        "informational_level": informational
    }


def get_user_inputs():
    """Get all user inputs including style preferences."""
    
    print("🤖 LinkedIn Slop Bot - Interactive Test")
    print("=" * 50)
    
    # Check API key
    if not os.getenv("NVIDIA_API_KEY"):
        print("❌ NVIDIA_API_KEY not found")
        print("Please set your NVIDIA API key as an environment variable")
        return None
    
    print("✅ NVIDIA_API_KEY is configured")
    
    # Collect style preferences first
    style_prefs = collect_style_preferences()
    
    # Image input
    print("\n📸 Image Input:")
    print("Enter image filename (should be in /home/ubuntu/build-an-agent/code/images directory)")
    print("Examples: image.jpeg, photo.png, pic.jpg")
    print("Or press Enter to skip image analysis")
    
    # Show available images
    images_dir = Path("images")
    if images_dir.exists():
        image_files = [f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']]
        if image_files:
            print(f"\n📁 Available images in {images_dir}:")
            for img in image_files:
                print(f"  - {img.name}")
    
    image_filename = input("\nImage filename: ").strip()
    
    image_path = None
    if image_filename:
        full_image_path = images_dir / image_filename
        if full_image_path.exists():
            image_path = str(full_image_path)
            print(f"✅ Found image: {image_path}")
        else:
            print(f"⚠️  Image not found: {full_image_path}")
            print("Continuing without image...")
    
    # Content prompt
    print("\n📝 Content Prompt:")
    print("What should the LinkedIn post be about?")
    print("Examples:")
    print("  - 'Create a motivational post about learning new skills'")
    print("  - 'Share insights about remote work productivity'")
    print("  - 'Tell a transformation story about career growth'")
    
    content_prompt = input("\nYour prompt: ").strip()
    if not content_prompt:
        content_prompt = "Write a general business post"
    
    # Post type
    print("\n🎭 Post Type:")
    print("1. general - Basic LinkedIn content")
    print("2. motivation - Motivational/inspirational posts")
    print("3. industry_insight - Industry-specific insights")
    print("4. personal_story - Personal transformation stories")
    
    post_type_input = input("Choose post type (1-4) or press Enter for 'general': ").strip()
    post_type_map = {
        "1": "general",
        "2": "motivation", 
        "3": "industry_insight",
        "4": "personal_story"
    }
    post_type = post_type_map.get(post_type_input, "general")
    
    return {
        "content_prompt": content_prompt,
        "image_path": image_path,
        "post_type": post_type,
        **style_prefs  # Add style preferences to the returned dict
    }


async def main():
    """Main function to run the LinkedIn Slop Bot."""
    
    parser = argparse.ArgumentParser(description="LinkedIn Slop Bot")
    parser.add_argument("--quick", action="store_true", help="Quick test mode (no image)")
    args = parser.parse_args()
    
    if args.quick:
        # Quick test mode with default style preferences
        print("⚡ Quick test mode - using defaults")
        inputs = {
            "content_prompt": "Write about the importance of teamwork in tech",
            "image_path": None,
            "post_type": "general",
            "grammar_level": 3,
            "emoji_level": 3,
            "hashtag_level": 3,
            "ragebait_level": 2,
            "inspirational_level": 3,
            "informational_level": 3
        }
    else:
        # Interactive mode
        inputs = get_user_inputs()
        if not inputs:
            return
    
    # Create directories
    Path("images").mkdir(exist_ok=True)
    Path("posts").mkdir(exist_ok=True)
    
    print("\n🚀 Generating LinkedIn post...")
    print(f"Prompt: {inputs['content_prompt']}")
    print(f"Image: {inputs['image_path'] or 'None'}")
    print(f"Type: {inputs['post_type']}")
    print(f"Style: Grammar={inputs['grammar_level']}, Emojis={inputs['emoji_level']}, Hashtags={inputs['hashtag_level']}, Ragebait={inputs['ragebait_level']}, Inspirational={inputs['inspirational_level']}, Informational={inputs['informational_level']}")
    print("-" * 50)
    
    try:
        # Generate the post with style preferences
        result = await async_create_linkedin_post(
            initial_prompt=inputs["content_prompt"],
            image_path=inputs["image_path"],
            post_type=inputs["post_type"],
            grammar_level=inputs["grammar_level"],
            emoji_level=inputs["emoji_level"],
            hashtag_level=inputs["hashtag_level"],
            ragebait_level=inputs["ragebait_level"],
            inspirational_level=inputs["inspirational_level"],
            informational_level=inputs["informational_level"]
        )
        
        if result and result.get("final_post"):
            print("\n🎯 GENERATED LINKEDIN POST:")
            print("=" * 50)
            print(result["final_post"])
            print("=" * 50)
            
            # Save to file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path("posts") / f"linkedin_post_{timestamp}.md"
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result["final_post"])
            
            print(f"✅ Saved to: {output_file}")
        else:
            print("❌ Failed to generate post")
            if result:
                print(f"Available keys: {list(result.keys())}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 