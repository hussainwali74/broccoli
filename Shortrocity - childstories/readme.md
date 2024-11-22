shortracity to create children stories

stories_gen.cursorrules consists of rules for the story generation using cursor .
stories_ideas.json consists of story ideas.
generated folder consists of generated stories in json format.

---
take the generated stories and create stories for children aged 12 and under.


-- 
Imager Module:
    description: This module is used to generate images from the text.
    files:
        img_prompter.py: will take the text and generate prompts for consistent images. Scenes, character descriptions, etc should be consistent from one story point to the next in a story.
        imagen.py: will take the prompts and generate images.

-------------
conda activate captacity
pip install -r requirements.txt
python script.py


Current:
    /data:
        has all the data new and generated.
        /generated_videos:
            has all the videos generated.
        /processed_data:
            has all the data processed.
            /done:
                has list of all the stories done.
                /story folder:
                    /images
                    /narrations
                    story_name.json
            /stories:
                has stories folder for each story. Before processing stories are in this folder.
                /story_name:
                    /images
                    /narrations
                    story_name.json
        /stories_data:
            /generated:
                has all the stories jsons data. These are yet to be processed and used to generate videos.
            /stories_ideas:
                has all the stories ideas. These will be used to generate stories jsons.

             
