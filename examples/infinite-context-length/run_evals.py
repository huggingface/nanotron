"""
Nanotron Inference Script

Usage:
```
export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=4 run_generate.py ---ckpt-path checkpoints/test/4
```
"""

import argparse
import os
from pathlib import Path

import torch
from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import (
    GenerationArgs,
    LoggingArgs,
    ParallelismArgs,
    get_config_from_file,
)
from nanotron.generation.decode import (
    GenerationInput,
    TokenizerConfig,
    decode_text,
)
from nanotron.logging import log_rank, set_ranks_logging_level
from nanotron.models import build_model
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import sanity_check
from nanotron.parallel.pipeline_parallel.engine import (
    OneForwardOneBackwardPipelineEngine,
)
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.random import (
    RandomStates,
    get_current_random_state,
    get_synced_random_state,
    set_random_seed,
)
from nanotron.serialize import load_weights
from nanotron.trainer import CONFIG_TO_MODEL_CLASS, mark_tied_parameters

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

logger = logging.get_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=Path, required=True, help="Checkpoint path")
    parser.add_argument("--dp", type=int, default=0)
    parser.add_argument("--pp", type=int, default=0)
    parser.add_argument("--tp", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=30, help="Maximum number of new tokens to generate")
    parser.add_argument("--context_length", type=int, required=True)
    parser.add_argument("--depth_percent", type=int, required=True)
    parser.add_argument("--num_shots", type=int, required=True)
    parser.add_argument("--num_digits", type=int, required=True)
    return parser.parse_args()


def generate(args, model, tokenizer, inputs, parallel_context) -> bool:
    outputs = decode_text(
        input_iter=(GenerationInput(text=text) for text in inputs),
        tokenizer=tokenizer,
        # TODO @thomasw21: From ModelWithLoss extract the model.
        model=model.model,
        parallel_context=parallel_context,
        max_new_tokens=args.max_new_tokens,
        max_micro_batch_size=1,
        generation_config=GenerationArgs(sampler="greedy", use_cache=False),
        tokenizer_config=TokenizerConfig(max_input_length=None),
        is_bench=os.environ.get("USE_BENCH", "0") == "1",
    )
    responses = []
    for output in outputs:
        input_ids = output.input_ids
        generated_ids = output.generation_ids
        answer_ids = generated_ids[len(input_ids) :]
        decoded_answer = tokenizer.decode(answer_ids, clean_up_tokenization_spaces=False)

        if isinstance(input_ids, TensorPointer):
            assert isinstance(generated_ids, TensorPointer)
            continue
        assert isinstance(generated_ids, torch.Tensor)

        log_rank(
            # f"input: {tokenizer.decode(input_ids, clean_up_tokenization_spaces=False)[:1000]}",
            f"input: {tokenizer.decode(input_ids, clean_up_tokenization_spaces=False)}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        log_rank(
            f"generation: {decoded_answer}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        log_rank(
            "--------------------------------------------------",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        responses.append(decoded_answer)

    return responses


HARRY_POTTER = """
Harry Potter and the Sorcerer's Stone


CHAPTER ONE

THE BOY WHO LIVED

Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say
that they were perfectly normal, thank you very much. They were the last
people you'd expect to be involved in anything strange or mysterious,
because they just didn't hold with such nonsense.

Mr. Dursley was the director of a firm called Grunnings, which made
drills. He was a big, beefy man with hardly any neck, although he did
have a very large mustache. Mrs. Dursley was thin and blonde and had
nearly twice the usual amount of neck, which came in very useful as she
spent so much of her time craning over garden fences, spying on the
neighbors. The Dursleys had a small son called Dudley and in their
opinion there was no finer boy anywhere.

The Dursleys had everything they wanted, but they also had a secret, and
their greatest fear was that somebody would discover it. They didn't
think they could bear it if anyone found out about the Potters. Mrs.
Potter was Mrs. Dursley's sister, but they hadn't met for several years;
in fact, Mrs. Dursley pretended she didn't have a sister, because her
sister and her good-for-nothing husband were as unDursleyish as it was
possible to be. The Dursleys shuddered to think what the neighbors would
say if the Potters arrived in the street. The Dursleys knew that the
Potters had a small son, too, but they had never even seen him. This boy
was another good reason for keeping the Potters away; they didn't want
Dudley mixing with a child like that.

When Mr. and Mrs. Dursley woke up on the dull, gray Tuesday our story
starts, there was nothing about the cloudy sky outside to suggest that
strange and mysterious things would soon be happening all over the
country. Mr. Dursley hummed as he picked out his most boring tie for
work, and Mrs. Dursley gossiped away happily as she wrestled a screaming
Dudley into his high chair.

None of them noticed a large, tawny owl flutter past the window.

At half past eight, Mr. Dursley picked up his briefcase, pecked Mrs.
Dursley on the cheek, and tried to kiss Dudley good-bye but missed,
because Dudley was now having a tantrum and throwing his cereal at the
walls. 'Little tyke,' chortled Mr. Dursley as he left the house. He got
into his car and backed out of number four's drive.

It was on the corner of the street that he noticed the first sign of
something peculiar -- a cat reading a map. For a second, Mr. Dursley
didn't realize what he had seen -- then he jerked his head around to
look again. There was a tabby cat standing on the corner of Privet
Drive, but there wasn't a map in sight. What could he have been thinking
of? It must have been a trick of the light. Mr. Dursley blinked and
stared at the cat. It stared back. As Mr. Dursley drove around the
corner and up the road, he watched the cat in his mirror. It was now
reading the sign that said Privet Drive -- no, looking at the sign; cats
couldn't read maps or signs. Mr. Dursley gave himself a little shake and
put the cat out of his mind. As he drove toward town he thought of
nothing except a large order of drills he was hoping to get that day.

But on the edge of town, drills were driven out of his mind by something
else. As he sat in the usual morning traffic jam, he couldn't help
noticing that there seemed to be a lot of strangely dressed people
about. People in cloaks. Mr. Dursley couldn't bear people who dressed in
funny clothes -- the getups you saw on young people! He supposed this
was some stupid new fashion. He drummed his fingers on the steering
wheel and his eyes fell on a huddle of these weirdos standing quite
close by. They were whispering excitedly together. Mr. Dursley was
enraged to see that a couple of them weren't young at all; why, that man
had to be older than he was, and wearing an emerald-green cloak! The
nerve of him! But then it struck Mr. Dursley that this was probably some
silly stunt -- these people were obviously collecting for something...
yes, that would be it. The traffic moved on and a few minutes later, Mr.
Dursley arrived in the Grunnings parking lot, his mind back on drills.

Mr. Dursley always sat with his back to the window in his office on the
ninth floor. If he hadn't, he might have found it harder to concentrate
on drills that morning. He didn't see the owls swoop ing past in broad
daylight, though people down in the street did; they pointed and gazed
open- mouthed as owl after owl sped overhead. Most of them had never
seen an owl even at nighttime. Mr. Dursley, however, had a perfectly
normal, owl-free morning. He yelled at five different people. He made
several important telephone calls and shouted a bit more. He was in a
very good mood until lunchtime, when he thought he'd stretch his legs
and walk across the road to buy himself a bun from the bakery.

He'd forgotten all about the people in cloaks until he passed a group of
them next to the baker's. He eyed them angrily as he passed. He didn't
know why, but they made him uneasy. This bunch were whispering
excitedly, too, and he couldn't see a single collecting tin. It was on
his way back past them, clutching a large doughnut in a bag, that he
caught a few words of what they were saying.

'The Potters, that's right, that's what I heard yes, their son, Harry'

Mr. Dursley stopped dead. Fear flooded him. He looked back at the
whisperers as if he wanted to say something to them, but thought better
of it.

He dashed back across the road, hurried up to his office, snapped at his
secretary not to disturb him, seized his telephone, and had almost
finished dialing his home number when he changed his mind. He put the
receiver back down and stroked his mustache, thinking... no, he was
being stupid. Potter wasn't such an unusual name. He was sure there were
lots of people called Potter who had a son called Harry. Come to think
of it, he wasn't even sure his nephew was called Harry. He'd never even
seen the boy. It might have been Harvey. Or Harold. There was no point
in worrying Mrs. Dursley; she always got so upset at any mention of her
sister. He didn't blame her -- if he'd had a sister like that... but all
the same, those people in cloaks...

He found it a lot harder to concentrate on drills that afternoon and
when he left the building at five o'clock, he was still so worried that
he walked straight into someone just outside the door.

'Sorry,' he grunted, as the tiny old man stumbled and almost fell. It
was a few seconds before Mr. Dursley realized that the man was wearing a
violet cloak. He didn't seem at all upset at being almost knocked to the
ground. On the contrary, his face split into a wide smile and he said in
a squeaky voice that made passersby stare, 'Don't be sorry, my dear sir,
for nothing could upset me today! Rejoice, for You-Know-Who has gone at
last! Even Muggles like yourself should be celebrating, this happy,
happy day!'

And the old man hugged Mr. Dursley around the middle and walked off.

Mr. Dursley stood rooted to the spot. He had been hugged by a complete
stranger. He also thought he had been called a Muggle, whatever that
was. He was rattled. He hurried to his car and set off for home, hoping
he was imagining things, which he had never hoped before, because he
didn't approve of imagination.

As he pulled into the driveway of number four, the first thing he saw --
and it didn't improve his mood -- was the tabby cat he'd spotted that
morning. It was now sitting on his garden wall. He was sure it was the
same one; it had the same markings around its eyes.

'Shoo!' said Mr. Dursley loudly. The cat didn't move. It just gave him a
stern look. Was this normal cat behavior? Mr. Dursley wondered. Trying
to pull himself together, he let himself into the house. He was still
determined not to mention anything to his wife.

Mrs. Dursley had had a nice, normal day. She told him over dinner all
about Mrs. Next Door's problems with her daughter and how Dudley had
learned a new word ('Won't!'). Mr. Dursley tried to act normally. When
Dudley had been put to bed, he went into the living room in time to
catch the last report on the evening news:

'And finally, bird-watchers everywhere have reported that the nation's
owls have been behaving very unusually today. Although owls normally
hunt at night and are hardly ever seen in daylight, there have been
hundreds of sightings of these birds flying in every direction since
sunrise. Experts are unable to explain why the owls have suddenly
changed their sleeping pattern.' The newscaster allowed himself a grin.
'Most mysterious. And now, over to Jim McGuffin with the weather. Going
to be any more showers of owls tonight, Jim?'

'Well, Ted,' said the weatherman, 'I don't know about that, but it's not
only the owls that have been acting oddly today. Viewers as far apart as
Kent, Yorkshire, and Dundee have been phoning in to tell me that instead
of the rain I promised yesterday, they've had a downpour of shooting
stars! Perhaps people have been celebrating Bonfire Night early -- it's
not until next week, folks! But I can promise a wet night tonight.'

Mr. Dursley sat frozen in his armchair. Shooting stars all over Britain?
Owls flying by daylight? Mysterious people in cloaks all over the place?
And a whisper, a whisper about the Potters...

Mrs. Dursley came into the living room carrying two cups of tea. It was
no good. He'd have to say something to her. He cleared his throat
nervously. 'Er -- Petunia, dear -- you haven't heard from your sister
lately, have you?'

As he had expected, Mrs. Dursley looked shocked and angry. After all,
they normally pretended she didn't have a sister.

'No,' she said sharply. 'Why?'

'Funny stuff on the news,' Mr. Dursley mumbled. 'Owls... shooting
stars... and there were a lot of funny-looking people in town today...'

'So?' snapped Mrs. Dursley.

'Well, I just thought... maybe... it was something to do with... you
know... her crowd.'

Mrs. Dursley sipped her tea through pursed lips. Mr. Dursley wondered
whether he dared tell her he'd heard the name 'Potter.' He decided he
didn't dare. Instead he said, as casually as he could, 'Their son --
he'd be about Dudley's age now, wouldn't he?'

'I suppose so,' said Mrs. Dursley stiffly.

'What's his name again? Howard, isn't it?'

'Harry. Nasty, common name, if you ask me.'

'Oh, yes,' said Mr. Dursley, his heart sinking horribly. 'Yes, I quite
agree.'

He didn't say another word on the subject as they went upstairs to bed.
While Mrs. Dursley was in the bathroom, Mr. Dursley crept to the bedroom
window and peered down into the front garden. The cat was still there.
It was staring down Privet Drive as though it were waiting for
something.

Was he imagining things? Could all this have anything to do with the
Potters? If it did... if it got out that they were related to a pair of
-- well, he didn't think he could bear it.

The Dursleys got into bed. Mrs. Dursley fell asleep quickly but Mr.
Dursley lay awake, turning it all over in his mind. His last, comforting
thought before he fell asleep was that even if the Potters were
involved, there was no reason for them to come near him and Mrs.
Dursley. The Potters knew very well what he and Petunia thought about
them and their kind.... He couldn't see how he and Petunia could get
mixed up in anything that might be going on -- he yawned and turned over
-- it couldn't affect them....

How very wrong he was.

Mr. Dursley might have been drifting into an uneasy sleep, but the cat
on the wall outside was showing no sign of sleepiness. It was sitting as
still as a statue, its eyes fixed unblinkingly on the far corner of
Privet Drive. It didn't so much as quiver when a car door slammed on the
next street, nor when two owls swooped overhead. In fact, it was nearly
midnight before the cat moved at all.

A man appeared on the corner the cat had been watching, appeared so
suddenly and silently you'd have thought he'd just popped out of the
ground. The cat's tail twitched and its eyes narrowed.

Nothing like this man had ever been seen on Privet Drive. He was tall,
thin, and very old, judging by the silver of his hair and beard, which
were both long enough to tuck into his belt. He was wearing long robes,
a purple cloak that swept the ground, and high-heeled, buckled boots.
His blue eyes were light, bright, and sparkling behind half-moon
spectacles and his nose was very long and crooked, as though it had been
broken at least twice. This man's name was Albus Dumbledore.

Albus Dumbledore didn't seem to realize that he had just arrived in a
street where everything from his name to his boots was unwelcome. He was
busy rummaging in his cloak, looking for something. But he did seem to
realize he was being watched, because he looked up suddenly at the cat,
which was still staring at him from the other end of the street. For
some reason, the sight of the cat seemed to amuse him. He chuckled and
muttered, 'I should have known.'

He found what he was looking for in his inside pocket. It seemed to be a
silver cigarette lighter. He flicked it open, held it up in the air, and
clicked it. The nearest street lamp went out with a little pop. He
clicked it again -- the next lamp flickered into darkness. Twelve times
he clicked the Put-Outer, until the only lights left on the whole street
were two tiny pinpricks in the distance, which were the eyes of the cat
watching him. If anyone looked out of their window now, even beady-eyed
Mrs. Dursley, they wouldn't be able to see anything that was happening
down on the pavement. Dumbledore slipped the Put-Outer back inside his
cloak and set off down the street toward number four, where he sat down
on the wall next to the cat. He didn't look at it, but after a moment he
spoke to it.

'Fancy seeing you here, Professor McGonagall.'

He turned to smile at the tabby, but it had gone. Instead he was smiling
at a rather severe-looking woman who was wearing square glasses exactly
the shape of the markings the cat had had around its eyes. She, too, was
wearing a cloak, an emerald one. Her black hair was drawn into a tight
bun. She looked distinctly ruffled.

'How did you know it was me?' she asked.

'My dear Professor, I 've never seen a cat sit so stiffly.'

'You'd be stiff if you'd been sitting on a brick wall all day,' said
Professor McGonagall.

'All day? When you could have been celebrating? I must have passed a
dozen feasts and parties on my way here.'

Professor McGonagall sniffed angrily.

'Oh yes, everyone's celebrating, all right,' she said impatiently.
'You'd think they'd be a bit more careful, but no -- even the Muggles
have noticed something's going on. It was on their news.' She jerked her
head back at the Dursleys' dark living-room window. 'I heard it. Flocks
of owls... shooting stars.... Well, they're not completely stupid. They
were bound to notice something. Shooting stars down in Kent -- I'll bet
that was Dedalus Diggle. He never had much sense.'

'You can't blame them,' said Dumbledore gently. 'We've had precious
little to celebrate for eleven years.'

'I know that,' said Professor McGonagall irritably. 'But that's no
reason to lose our heads. People are being downright careless, out on
the streets in broad daylight, not even dressed in Muggle clothes,
swapping rumors.'

She threw a sharp, sideways glance at Dumbledore here, as though hoping
he was going to tell her something, but he didn't, so she went on. 'A
fine thing it would be if, on the very day YouKnow-Who seems to have
disappeared at last, the Muggles found out about us all. I suppose he
really has gone, Dumbledore?'

'It certainly seems so,' said Dumbledore. 'We have much to be thankful
for. Would you care for a lemon drop?'

'A what?'

'A lemon drop. They're a kind of Muggle sweet I'm rather fond of'

'No, thank you,' said Professor McGonagall coldly, as though she didn't
think this was the moment for lemon drops. 'As I say, even if
You-Know-Who has gone -'

'My dear Professor, surely a sensible person like yourself can call him
by his name? All this 'You- Know-Who' nonsense -- for eleven years I
have been trying to persuade people to call him by his proper name:
Voldemort.' Professor McGonagall flinched, but Dumbledore, who was
unsticking two lemon drops, seemed not to notice. 'It all gets so
confusing if we keep saying 'You-Know-Who.' I have never seen any reason
to be frightened of saying Voldemort's name.

'I know you haven 't, said Professor McGonagall, sounding half
exasperated, half admiring. 'But you're different. Everyone knows you're
the only one You-Know- oh, all right, Voldemort, was frightened of.'

'You flatter me,' said Dumbledore calmly. 'Voldemort had powers I will
never have.'

'Only because you're too -- well -- noble to use them.'

'It's lucky it's dark. I haven't blushed so much since Madam Pomfrey
told me she liked my new earmuffs.'

Professor McGonagall shot a sharp look at Dumbledore and said, 'The owls
are nothing next to the rumors that are flying around. You know what
everyone's saying? About why he's disappeared? About what finally
stopped him?'

It seemed that Professor McGonagall had reached the point she was most
anxious to discuss, the real reason she had been waiting on a cold, hard
wall all day, for neither as a cat nor as a woman had she fixed
Dumbledore with such a piercing stare as she did now. It was plain that
whatever 'everyone' was saying, she was not going to believe it until
Dumbledore told her it was true. Dumbledore, however, was choosing
another lemon drop and did not answer.

'What they're saying,' she pressed on, 'is that last night Voldemort
turned up in Godric's Hollow. He went to find the Potters. The rumor is
that Lily and James Potter are -- are -- that they're -- dead. '

Dumbledore bowed his head. Professor McGonagall gasped.

'Lily and James... I can't believe it... I didn't want to believe it...
Oh, Albus...'

Dumbledore reached out and patted her on the shoulder. 'I know... I
know...' he said heavily.

Professor McGonagall's voice trembled as she went on. 'That's not all.
They're saying he tried to kill the Potter's son, Harry. But -- he
couldn't. He couldn't kill that little boy. No one knows why, or how,
but they're saying that when he couldn't kill Harry Potter, Voldemort's
power somehow broke -- and that's why he's gone.

Dumbledore nodded glumly.

'It's -- it's true?' faltered Professor McGonagall. 'After all he's
done... all the people he's killed... he couldn't kill a little boy?
It's just astounding... of all the things to stop him... but how in the
name of heaven did Harry survive?'

'We can only guess,' said Dumbledore. 'We may never know.'

Professor McGonagall pulled out a lace handkerchief and dabbed at her
eyes beneath her spectacles. Dumbledore gave a great sniff as he took a
golden watch from his pocket and examined it. It was a very odd watch.
It had twelve hands but no numbers; instead, little planets were moving
around the edge. It must have made sense to Dumbledore, though, because
he put it back in his pocket and said, 'Hagrid's late. I suppose it was
he who told you I'd be here, by the way?'

'Yes,' said Professor McGonagall. 'And I don't suppose you're going to
tell me why you're here, of all places?'

'I've come to bring Harry to his aunt and uncle. They're the only family
he has left now.'

'You don't mean -- you can't mean the people who live here?' cried
Professor McGonagall, jumping to her feet and pointing at number four.
'Dumbledore -- you can't. I've been watching them all day. You couldn't
find two people who are less like us. And they've got this son -- I saw
him kicking his mother all the way up the street, screaming for sweets.
Harry Potter come and live here!'

'It's the best place for him,' said Dumbledore firmly. 'His aunt and
uncle will be able to explain everything to him when he's older. I've
written them a letter.'

'A letter?' repeated Professor McGonagall faintly, sitting back down on
the wall. 'Really, Dumbledore, you think you can explain all this in a
letter? These people will never understand him! He'll be famous -- a
legend -- I wouldn't be surprised if today was known as Harry Potter day
in the future -- there will be books written about Harry -- every child
in our world will know his name!'

'Exactly,' said Dumbledore, looking very seriously over the top of his
half-moon glasses. 'It would be enough to turn any boy's head. Famous
before he can walk and talk! Famous for something he won't even
remember! CarA you see how much better off he'll be, growing up away
from all that until he's ready to take it?'

Professor McGonagall opened her mouth, changed her mind, swallowed, and
then said, 'Yes -- yes, you're right, of course. But how is the boy
getting here, Dumbledore?' She eyed his cloak suddenly as though she
thought he might be hiding Harry underneath it.

'Hagrid's bringing him.'

'You think it -- wise -- to trust Hagrid with something as important as
this?'

I would trust Hagrid with my life,' said Dumbledore.

'I'm not saying his heart isn't in the right place,' said Professor
McGonagall grudgingly, 'but you can't pretend he's not careless. He does
tend to -- what was that?'

A low rumbling sound had broken the silence around them. It grew
steadily louder as they looked up and down the street for some sign of a
headlight; it swelled to a roar as they both looked up at the sky -- and
a huge motorcycle fell out of the air and landed on the road in front of
them.

If the motorcycle was huge, it was nothing to the man sitting astride
it. He was almost twice as tall as a normal man and at least five times
as wide. He looked simply too big to be allowed, and so wild - long
tangles of bushy black hair and beard hid most of his face, he had hands
the size of trash can lids, and his feet in their leather boots were
like baby dolphins. In his vast, muscular arms he was holding a bundle
of blankets.

'Hagrid,' said Dumbledore, sounding relieved. 'At last. And where did
you get that motorcycle?'

'Borrowed it, Professor Dumbledore, sit,' said the giant, climbing
carefully off the motorcycle as he spoke. 'Young Sirius Black lent it to
me. I've got him, sir.'

'No problems, were there?'

'No, sir -- house was almost destroyed, but I got him out all right
before the Muggles started swarmin' around. He fell asleep as we was
flyin' over Bristol.'

Dumbledore and Professor McGonagall bent forward over the bundle of
blankets. Inside, just visible, was a baby boy, fast asleep. Under a
tuft of jet-black hair over his forehead they could see a curiously
shaped cut, like a bolt of lightning.

'Is that where -?' whispered Professor McGonagall.

'Yes,' said Dumbledore. 'He'll have that scar forever.'

'Couldn't you do something about it, Dumbledore?'

'Even if I could, I wouldn't. Scars can come in handy. I have one myself
above my left knee that is a perfect map of the London Underground. Well
-- give him here, Hagrid -- we'd better get this over with.'

Dumbledore took Harry in his arms and turned toward the Dursleys' house.

'Could I -- could I say good-bye to him, sir?' asked Hagrid. He bent his
great, shaggy head over Harry and gave him what must have been a very
scratchy, whiskery kiss. Then, suddenly, Hagrid let out a howl like a
wounded dog.

'Shhh!' hissed Professor McGonagall, 'you'll wake the Muggles!'

'S-s-sorry,' sobbed Hagrid, taking out a large, spotted handkerchief and
burying his face in it. 'But I c-c-can't stand it -- Lily an' James dead
-- an' poor little Harry off ter live with Muggles -'

'Yes, yes, it's all very sad, but get a grip on yourself, Hagrid, or
we'll be found,' Professor McGonagall whispered, patting Hagrid gingerly
on the arm as Dumbledore stepped over the low garden wall and walked to
the front door. He laid Harry gently on the doorstep, took a letter out
of his cloak, tucked it inside Harry's blankets, and then came back to
the other two. For a full minute the three of them stood and looked at
the little bundle; Hagrid's shoulders shook, Professor McGonagall
blinked furiously, and the twinkling light that usually shone from
Dumbledore's eyes seemed to have gone out.

'Well,' said Dumbledore finally, 'that's that. We've no business staying
here. We may as well go and join the celebrations.'

'Yeah,' said Hagrid in a very muffled voice, 'I'll be takin' Sirius his
bike back. G'night, Professor McGonagall -- Professor Dumbledore, sir.'

Wiping his streaming eyes on his jacket sleeve, Hagrid swung himself
onto the motorcycle and kicked the engine into life; with a roar it rose
into the air and off into the night.

'I shall see you soon, I expect, Professor McGonagall,' said Dumbledore,
nodding to her. Professor McGonagall blew her nose in reply.

Dumbledore turned and walked back down the street. On the corner he
stopped and took out the silver Put-Outer. He clicked it once, and
twelve balls of light sped back to their street lamps so that Privet
Drive glowed suddenly orange and he could make out a tabby cat slinking
around the corner at the other end of the street. He could just see the
bundle of blankets on the step of number four.

'Good luck, Harry,' he murmured. He turned on his heel and with a swish
of his cloak, he was gone.

A breeze ruffled the neat hedges of Privet Drive, which lay silent and
tidy under the inky sky, the very last place you would expect
astonishing things to happen. Harry Potter rolled over inside his
blankets without waking up. One small hand closed on the letter beside
him and he slept on, not knowing he was special, not knowing he was
famous, not knowing he would be woken in a few hours' time by Mrs.
Dursley's scream as she opened the front door to put out the milk
bottles, nor that he would spend the next few weeks being prodded and
pinched by his cousin Dudley... He couldn't know that at this very
moment, people meeting in secret all over the country were holding up
their glasses and saying in hushed voices: 'To Harry Potter -- the boy
who lived!'


CHAPTER TWO

THE VANISHING GLASS

Nearly ten years had passed since the Dursleys had woken up to find
their nephew on the front step, but Privet Drive had hardly changed at
all. The sun rose on the same tidy front gardens and lit up the brass
number four on the Dursleys' front door; it crept into their living
room, which was almost exactly the same as it had been on the night when
Mr. Dursley had seen that fateful news report about the owls. Only the
photographs on the mantelpiece really showed how much time had passed.
Ten years ago, there had been lots of pictures of what looked like a
large pink beach ball wearing different-colored bonnets -- but Dudley
Dursley was no longer a baby, and now the photographs showed a large
blond boy riding his first bicycle, on a carousel at the fair, playing a
computer game with his father, being hugged and kissed by his mother.
The room held no sign at all that another boy lived in the house, too.

Yet Harry Potter was still there, asleep at the moment, but not for
long. His Aunt Petunia was awake and it was her shrill voice that made
the first noise of the day.

'Up! Get up! Now!'

Harry woke with a start. His aunt rapped on the door again.

'Up!' she screeched. Harry heard her walking toward the kitchen and then
the sound of the frying pan being put on the stove. He rolled onto his
back and tried to remember the dream he had been having. It had been a
good one. There had been a flying motorcycle in it. He had a funny
feeling he'd had the same dream before.

His aunt was back outside the door.

'Are you up yet?' she demanded.

'Nearly,' said Harry.

'Well, get a move on, I want you to look after the bacon. And don't you
dare let it burn, I want everything perfect on Duddy's birthday.'

Harry groaned.

'What did you say?' his aunt snapped through the door.

'Nothing, nothing...'

Dudley's birthday -- how could he have forgotten? Harry got slowly out
of bed and started looking for socks. He found a pair under his bed and,
after pulling a spider off one of them, put them on. Harry was used to
spiders, because the cupboard under the stairs was full of them, and
that was where he slept.

When he was dressed he went down the hall into the kitchen. The table
was almost hidden beneath all Dudley's birthday presents. It looked as
though Dudley had gotten the new computer he wanted, not to mention the
second television and the racing bike. Exactly why Dudley wanted a
racing bike was a mystery to Harry, as Dudley was very fat and hated
exercise -- unless of course it involved punching somebody. Dudley's
favorite punching bag was Harry, but he couldn't often catch him. Harry
didn't look it, but he was very fast.

Perhaps it had something to do with living in a dark cupboard, but Harry
had always been small and skinny for his age. He looked even smaller and
skinnier than he really was because all he had to wear were old clothes
of Dudley's, and Dudley was about four times bigger than he was. Harry
had a thin face, knobbly knees, black hair, and bright green eyes. He
wore round glasses held together with a lot of Scotch tape because of
all the times Dudley had punched him on the nose. The only thing Harry
liked about his own appearance was a very thin scar on his forehead that
was shaped like a bolt of lightning. He had had it as long as he could
remember, and the first question he could ever remember asking his Aunt
Petunia was how he had gotten it.

'In the car crash when your parents died,' she had said. 'And don't ask
questions.'

Don't ask questions -- that was the first rule for a quiet life with the
Dursleys.

Uncle Vernon entered the kitchen as Harry was turning over the bacon.

'Comb your hair!' he barked, by way of a morning greeting.

About once a week, Uncle Vernon looked over the top of his newspaper and
shouted that Harry needed a haircut. Harry must have had more haircuts
than the rest of the boys in his class put

together, but it made no difference, his hair simply grew that way --
all over the place.

Harry was frying eggs by the time Dudley arrived in the kitchen with his
mother. Dudley looked a lot like Uncle Vernon. He had a large pink face,
not much neck, small, watery blue eyes, and thick blond hair that lay
smoothly on his thick, fat head. Aunt Petunia often said that Dudley
looked like a baby angel -- Harry often said that Dudley looked like a
pig in a wig.

Harry put the plates of egg and bacon on the table, which was difficult
as there wasn't much room. Dudley, meanwhile, was counting his presents.
His face fell.

'Thirty-six,' he said, looking up at his mother and father. 'That's two
less than last year.'

'Darling, you haven't counted Auntie Marge's present, see, it's here
under this big one from Mommy and Daddy.'

'All right, thirty-seven then,' said Dudley, going red in the face.
Harry, who could see a huge Dudley tantrum coming on, began wolfing down
his bacon as fast as possible in case Dudley turned the table over.

Aunt Petunia obviously scented danger, too, because she said quickly,
'And we'll buy you another two presents while we're out today. How's
that, popkin? Two more presents. Is that all right''

Dudley thought for a moment. It looked like hard work. Finally he said
slowly, 'So I'll have thirty ... thirty...'

'Thirty-nine, sweetums,' said Aunt Petunia.

'Oh.' Dudley sat down heavily and grabbed the nearest parcel. 'All right
then.'

Uncle Vernon chuckled. 'Little tyke wants his money's worth, just like
his father. 'Atta boy, Dudley!' He ruffled Dudley's hair.

At that moment the telephone rang and Aunt Petunia went to answer it
while Harry and Uncle Vernon watched Dudley unwrap the racing bike, a
video camera, a remote control airplane, sixteen new computer games, and
a VCR. He was ripping the paper off a gold wristwatch when Aunt Petunia
came back from the telephone looking both angry and worried.

'Bad news, Vernon,' she said. 'Mrs. Figg's broken her leg. She can't
take him.' She jerked her head in Harry's direction.

Dudley's mouth fell open in horror, but Harry's heart gave a leap. Every
year on Dudley's birthday, his parents took him and a friend out for the
day, to adventure parks, hamburger restaurants, or the movies. Every
year, Harry was left behind with Mrs. Figg, a mad old lady who lived two
streets away. Harry hated it there. The whole house smelled of cabbage
and Mrs. Figg made him look at photographs of all the cats she'd ever
owned.

'Now what?' said Aunt Petunia, looking furiously at Harry as though he'd
planned this. Harry knew he ought to feel sorry that Mrs. Figg had
broken her leg, but it wasn't easy when he reminded himself it would be
a whole year before he had to look at Tibbles, Snowy, Mr. Paws, and
Tufty again.

'We could phone Merge,' Uncle Vernon suggested.

'Don't be silly, Vernon, she hates the boy.'

The Dursleys often spoke about Harry like this, as though he wasn't
there -- or rather, as though he was something very nasty that couldn't
understand them, like a slug.

'What about what's-her-name, your friend -- Yvonne?'

'On vacation in Majorca,' snapped Aunt Petunia.

'You could just leave me here,' Harry put in hopefully (he'd be able to
watch what he wanted on television for a change and maybe even have a go
on Dudley's computer).

Aunt Petunia looked as though she'd just swallowed a lemon.

'And come back and find the house in ruins?' she snarled.

'I won't blow up the house,' said Harry, but they weren't listening.

'I suppose we could take him to the zoo,' said Aunt Petunia slowly, '...
and leave him in the car....'

'That car's new, he's not sitting in it alone....'

Dudley began to cry loudly. In fact, he wasn't really crying -- it had
been years since he'd really cried -- but he knew that if he screwed up
his face and wailed, his mother would give him anything he wanted.

'Dinky Duddydums, don't cry, Mummy won't let him spoil your special
day!' she cried, flinging her arms around him.

'I... don't... want... him... t-t-to come!' Dudley yelled between huge,
pretend sobs. 'He always sp- spoils everything!' He shot Harry a nasty
grin through the gap in his mother's arms.

Just then, the doorbell rang -- 'Oh, good Lord, they're here!' said Aunt
Petunia frantically -- and a moment later, Dudley's best friend, Piers
Polkiss, walked in with his mother. Piers was a scrawny boy with a face
like a rat. He was usually the one who held people's arms behind their
backs while Dudley hit them. Dudley stopped pretending to cry at once.

Half an hour later, Harry, who couldn't believe his luck, was sitting in
the back of the Dursleys' car with Piers and Dudley, on the way to the
zoo for the first time in his life. His aunt and uncle hadn't been able
to think of anything else to do with him, but before they'd left, Uncle
Vernon had taken Harry aside.

'I'm warning you,' he had said, putting his large purple face right up
close to Harry's, 'I'm warning you now, boy -- any funny business,
anything at all -- and you'll be in that cupboard from now until
Christmas.'

'I'm not going to do anything,' said Harry, 'honestly..

But Uncle Vernon didn't believe him. No one ever did.

The problem was, strange things often happened around Harry and it was
just no good telling the Dursleys he didn't make them happen.

Once, Aunt Petunia, tired of Harry coming back from the barbers looking
as though he hadn't been at all, had taken a pair of kitchen scissors
and cut his hair so short he was almost bald except for his bangs, which
she left 'to hide that horrible scar.' Dudley had laughed himself silly
at Harry, who spent a sleepless night imagining school the next day,
where he was already laughed at for his baggy clothes and taped glasses.
Next morning, however, he had gotten up to find his hair exactly as it
had been before Aunt Petunia had sheared it off He had been given a week
in his cupboard for this, even though he had tried to explain that he
couldn't explain how it had grown back so quickly.

Another time, Aunt Petunia had been trying to force him into a revolting
old sweater of Dudley's (brown with orange puff balls) -- The harder she
tried to pull it over his head, the smaller it seemed to become, until
finally it might have fitted a hand puppet, but certainly wouldn't fit
Harry. Aunt Petunia had decided it must have shrunk in the wash and, to
his great relief, Harry wasn't punished.

On the other hand, he'd gotten into terrible trouble for being found on
the roof of the school kitchens. Dudley's gang had been chasing him as
usual when, as much to Harry's surprise as anyone else's, there he was
sitting on the chimney. The Dursleys had received a very angry letter
from Harry's headmistress telling them Harry had been climbing school
buildings. But all he'd tried to do (as he shouted at Uncle Vernon
through the locked door of his cupboard) was jump behind the big trash
cans outside the kitchen doors. Harry supposed that the wind must have
caught him in mid- jump.

But today, nothing was going to go wrong. It was even worth being with
Dudley and Piers to be spending the day somewhere that wasn't school,
his cupboard, or Mrs. Figg's cabbage-smelling living room.

While he drove, Uncle Vernon complained to Aunt Petunia. He liked to
complain about things: people at work, Harry, the council, Harry, the
bank, and Harry were just a few of his favorite subjects. This morning,
it was motorcycles.

'... roaring along like maniacs, the young hoodlums,' he said, as a
motorcycle overtook them.

I had a dream about a motorcycle,' said Harry, remembering suddenly. 'It
was flying.'

Uncle Vernon nearly crashed into the car in front. He turned right
around in his seat and yelled at Harry, his face like a gigantic beet
with a mustache: 'MOTORCYCLES DON'T FLY!'

Dudley and Piers sniggered.

I know they don't,' said Harry. 'It was only a dream.'

But he wished he hadn't said anything. If there was one thing the
Dursleys hated even more than his asking questions, it was his talking
about anything acting in a way it shouldn't, no matter if it was in a
dream or even a cartoon -- they seemed to think he might get dangerous
ideas.

It was a very sunny Saturday and the zoo was crowded with families. The
Dursleys bought Dudley and Piers large chocolate ice creams at the
entrance and then, because the smiling lady in the van had asked Harry
what he wanted before they could hurry him away, they bought him a cheap
lemon ice pop. It wasn't bad, either, Harry thought, licking it as they
watched a gorilla scratching its head who looked remarkably like Dudley,
except that it wasn't blond.

Harry had the best morning he'd had in a long time. He was careful to
walk a little way apart from the Dursleys so that Dudley and Piers, who
were starting to get bored with the animals by lunchtime, wouldn't fall
back on their favorite hobby of hitting him. They ate in the zoo
restaurant, and when Dudley had a tantrum because his knickerbocker
glory didn't have enough ice cream on top, Uncle Vernon bought him
another one and Harry was allowed to finish the first.

Harry felt, afterward, that he should have known it was all too good to
last.

After lunch they went to the reptile house. It was cool and dark in
there, with lit windows all along the walls. Behind the glass, all sorts
of lizards and snakes were crawling and slithering over bits of wood and
stone. Dudley and Piers wanted to see huge, poisonous cobras and thick,
man-crushing pythons. Dudley quickly found the largest snake in the
place. It could have wrapped its body twice around Uncle Vernon's car
and crushed it into a trash can -- but at the moment it didn't look in
the mood. In fact, it was fast asleep.

Dudley stood with his nose pressed against the glass, staring at the
glistening brown coils.

'Make it move,' he whined at his father. Uncle Vernon tapped on the
glass, but the snake didn't budge.

'Do it again,' Dudley ordered. Uncle Vernon rapped the glass smartly
with his knuckles, but the snake just snoozed on.

'This is boring,' Dudley moaned. He shuffled away.

Harry moved in front of the tank and looked intently at the snake. He
wouldn't have been surprised if it had died of boredom itself -- no
company except stupid people drumming their fingers on the glass trying
to disturb it all day long. It was worse than having a cupboard as a
bedroom, where the only visitor was Aunt Petunia hammering on the door
to wake you up; at least he got to visit the rest of the house.

The snake suddenly opened its beady eyes. Slowly, very slowly, it raised
its head until its eyes were on a level with Harry's.

It winked.

Harry stared. Then he looked quickly around to see if anyone was
watching. They weren't. He looked back at the snake and winked, too.

The snake jerked its head toward Uncle Vernon and Dudley, then raised
its eyes to the ceiling. It gave Harry a look that said quite plainly:

'I get that all the time.

'I know,' Harry murmured through the glass, though he wasn't sure the
snake could hear him. 'It must be really annoying.'

The snake nodded vigorously.

'Where do you come from, anyway?' Harry asked.

The snake jabbed its tail at a little sign next to the glass. Harry
peered at it.

Boa Constrictor, Brazil.

'Was it nice there?'

The boa constrictor jabbed its tail at the sign again and Harry read on:
This specimen was bred in the zoo. 'Oh, I see -- so you've never been to
Brazil?'

As the snake shook its head, a deafening shout behind Harry made both of
them jump.

'DUDLEY! MR. DURSLEY! COME AND LOOK AT THIS SNAKE! YOU WON'T BELIEVE
WHAT IT'S DOING!'

Dudley came waddling toward them as fast as he could.

'Out of the way, you,' he said, punching Harry in the ribs. Caught by
surprise, Harry fell hard on the concrete floor. What came next happened
so fast no one saw how it happened -- one second, Piers and Dudley were
leaning right up close to the glass, the next, they had leapt back with
howls of horror.

Harry sat up and gasped; the glass front of the boa constrictor's tank
had vanished. The great snake was uncoiling itself rapidly, slithering
out onto the floor. People throughout the reptile house screamed and
started running for the exits.

As the snake slid swiftly past him, Harry could have sworn a low,
hissing voice said, 'Brazil, here I come.... Thanksss, amigo.'

The keeper of the reptile house was in shock.

'But the glass,' he kept saying, 'where did the glass go?'

The zoo director himself made Aunt Petunia a cup of strong, sweet tea
while he apologized over and over again. Piers and Dudley could only
gibber. As far as Harry had seen, the snake hadn't done anything except
snap playfully at their heels as it passed, but by the time they were
all back in Uncle Vernon's car, Dudley was telling them how it had
nearly bitten off his leg, while Piers was swearing it had tried to
squeeze him to death. But worst of all, for Harry at least, was Piers
calming down enough to say, 'Harry was talking to it, weren't you,
Harry?'

Uncle Vernon waited until Piers was safely out of the house before
starting on Harry. He was so angry he could hardly speak. He managed to
say, 'Go -- cupboard -- stay -- no meals,' before he collapsed into a
chair, and Aunt Petunia had to run and get him a large brandy.

Harry lay in his dark cupboard much later, wishing he had a watch. He
didn't know what time it was and he couldn't be sure the Dursleys were
asleep yet. Until they were, he couldn't risk sneaking to the kitchen
for some food.

He'd lived with the Dursleys almost ten years, ten miserable years, as
long as he could remember, ever since he'd been a baby and his parents
had died in that car crash. He couldn't remember being in the car when
his parents had died. Sometimes, when he strained his memory during long
hours in his cupboard, he came up with a strange vision: a blinding
flash of green light and a burn- ing pain on his forehead. This, he
supposed, was the crash, though he couldn't imagine where all the green
light came from. He couldn't remember his parents at all. His aunt and
uncle never spoke about them, and of course he was forbidden to ask
questions. There were no photographs of them in the house.

When he had been younger, Harry had dreamed and dreamed of some unknown
relation coming to take him away, but it had never happened; the
Dursleys were his only family. Yet sometimes he thought (or maybe hoped)
that strangers in the street seemed to know him. Very strange strangers
they were, too. A tiny man in a violet top hat had bowed to him once
while out shopping with Aunt Petunia and Dudley. After asking Harry
furiously if he knew the man, Aunt Petunia had rushed them out of the
shop without buying anything. A wild-looking old woman dressed all in
green had waved merrily at him once on a bus. A bald man in a very long
purple coat had actually shaken his hand in the street the other day and
then walked away without a word. The weirdest thing about all these
people was the way they seemed to vanish the second Harry tried to get a
closer look.

At school, Harry had no one. Everybody knew that Dudley's gang hated
that odd Harry Potter in his baggy old clothes and broken glasses, and
nobody liked to disagree with Dudley's gang.


CHAPTER THREE

THE LETTERS FROM NO ONE

The escape of the Brazilian boa constrictor earned Harry his
longest-ever punishment. By the time he was allowed out of his cupboard
again, the summer holidays had started and Dudley had already broken his
new video camera, crashed his remote control airplane, and, first time
out on his racing bike, knocked down old Mrs. Figg as she crossed Privet
Drive on her crutches.

Harry was glad school was over, but there was no escaping Dudley's gang,
who visited the house every single day. Piers, Dennis, Malcolm, and
Gordon were all big and stupid, but as Dudley was the biggest and
stupidest of the lot, he was the leader. The rest of them were all quite
happy to join in Dudley's favorite sport: Harry Hunting.

This was why Harry spent as much time as possible out of the house,
wandering around and thinking about the end of the holidays, where he
could see a tiny ray of hope. When September came he would be going off
to secondary school and, for the first time in his life, he wouldn't be
with Dudley. Dudley had been accepted at Uncle Vernon's old private
school, Smeltings. Piers Polkiss was going there too. Harry, on the
other hand, was going to Stonewall High, the local public school. Dudley
thought this was very funny.

'They stuff people's heads down the toilet the first day at Stonewall,'
he told Harry. 'Want to come upstairs and practice?'

'No, thanks,' said Harry. 'The poor toilet's never had anything as
horrible as your head down it -- it might be sick.' Then he ran, before
Dudley could work out what he'd said.

One day in July, Aunt Petunia took Dudley to London to buy his Smeltings
uniform, leaving Harry at Mrs. Figg's. Mrs. Figg wasn 't as bad as
usual. It turned out she'd broken her leg tripping over one of her cats,
and she didn't seem quite as fond of them as before. She let Harry watch
television and gave him a bit of chocolate cake that tasted as though
she'd had it for several years.

That evening, Dudley paraded around the living room for the family in
his brand-new uniform. Smeltings' boys wore maroon tailcoats, orange
knickerbockers, and flat straw hats called boaters. They also carried
knobbly sticks, used for hitting each other while the teachers weren't
looking. This was supposed to be good training for later life.

As he looked at Dudley in his new knickerbockers, Uncle Vernon said
gruffly that it was the proudest moment of his life. Aunt Petunia burst
into tears and said she couldn't believe it was her Ickle Dudleykins, he
looked so handsome and grown-up. Harry didn't trust himself to speak. He
thought two of his ribs might already have cracked from trying not to
laugh.

There was a horrible smell in the kitchen the next morning when Harry
went in for breakfast. It seemed to be coming from a large metal tub in
the sink. He went to have a look. The tub was full of what looked like
dirty rags swimming in gray water.

'What's this?' he asked Aunt Petunia. Her lips tightened as they always
did if he dared to ask a question.

'Your new school uniform,' she said.

Harry looked in the bowl again.

'Oh,' he said, 'I didn't realize it had to be so wet.'

'DotA be stupid,' snapped Aunt Petunia. 'I'm dyeing some of Dudley's old
things gray for you. It'll look just like everyone else's when I've
finished.'

Harry seriously doubted this, but thought it best not to argue. He sat
down at the table and tried not to think about how he was going to look
on his first day at Stonewall High -- like he was wearing bits of old
elephant skin, probably.

Dudley and Uncle Vernon came in, both with wrinkled noses because of the
smell from Harry's new uniform. Uncle Vernon opened his newspaper as
usual and Dudley banged his Smelting stick, which he carried everywhere,
on the table.

They heard the click of the mail slot and flop of letters on the
doormat.

'Get the mail, Dudley,' said Uncle Vernon from behind his paper.

'Make Harry get it.'

'Get the mail, Harry.'

'Make Dudley get it.'

'Poke him with your Smelting stick, Dudley.'

Harry dodged the Smelting stick and went to get the mail. Three things
lay on the doormat: a postcard from Uncle Vernon's sister Merge, who was
vacationing on the Isle of Wight, a brown envelope that looked like a
bill, and -- a letter for Harry.

Harry picked it up and stared at it, his heart twanging like a giant
elastic band. No one, ever, in his whole life, had written to him. Who
would? He had no friends, no other relatives -- he didn't belong to the
library, so he'd never even got rude notes asking for books back. Yet
here it was, a letter, addressed so plainly there could be no mistake:

Mr. H. Potter

The Cupboard under the Stairs

4 Privet Drive

Little Whinging

Surrey

The envelope was thick and heavy, made of yellowish parchment, and the
address was written in emerald-green ink. There was no stamp.

Turning the envelope over, his hand trembling, Harry saw a purple wax
seal bearing a coat of arms; a lion, an eagle, a badger, and a snake
surrounding a large letter H.

'Hurry up, boy!' shouted Uncle Vernon from the kitchen. 'What are you
doing, checking for letter bombs?' He chuckled at his own joke.

Harry went back to the kitchen, still staring at his letter. He handed
Uncle Vernon the bill and the postcard, sat down, and slowly began to
open the yellow envelope.

Uncle Vernon ripped open the bill, snorted in disgust, and flipped over
the postcard.

'Marge's ill,' he informed Aunt Petunia. 'Ate a funny whelk. --.'

'Dad!' said Dudley suddenly. 'Dad, Harry's got something!'

Harry was on the point of unfolding his letter, which was written on the
same heavy parchment as the envelope, when it was jerked sharply out of
his hand by Uncle Vernon.

'That's mine!' said Harry, trying to snatch it back.

'Who'd be writing to you?' sneered Uncle Vernon, shaking the letter open
with one hand and glancing at it. His face went from red to green faster
than a set of traffic lights. And it didn't stop there. Within seconds
it was the grayish white of old porridge.

'P-P-Petunia!' he gasped.

Dudley tried to grab the letter to read it, but Uncle Vernon held it
high out of his reach. Aunt Petunia took it curiously and read the first
line. For a moment it looked as though she might faint. She clutched her
throat and made a choking noise.

'Vernon! Oh my goodness -- Vernon!'

They stared at each other, seeming to have forgotten that Harry and
Dudley were still in the room. Dudley wasn't used to being ignored. He
gave his father a sharp tap on the head with his Smelting stick.

'I want to read that letter,' he said loudly. want to read it,' said
Harry furiously, 'as it's mine.'

'Get out, both of you,' croaked Uncle Vernon, stuffing the letter back
inside its envelope.

Harry didn't move.

I WANT MY LETTER!' he shouted.

'Let me see it!' demanded Dudley.

'OUT!' roared Uncle Vernon, and he took both Harry and Dudley by the
scruffs of their necks and threw them into the hall, slamming the
kitchen door behind them. Harry and Dudley promptly had a furious but
silent fight over who would listen at the keyhole; Dudley won, so Harry,
his glasses dangling from one ear, lay flat on his stomach to listen at
the crack between door and floor.

'Vernon,' Aunt Petunia was saying in a quivering voice, 'look at the
address -- how could they possibly know where he sleeps? You don't think
they're watching the house?'

'Watching -- spying -- might be following us,' muttered Uncle Vernon
wildly.

'But what should we do, Vernon? Should we write back? Tell them we don't
want --'

Harry could see Uncle Vernon's shiny black shoes pacing up and down the
kitchen.

'No,' he said finally. 'No, we'll ignore it. If they don't get an
answer... Yes, that's best... we won't do anything....

'But --'

'I'm not having one in the house, Petunia! Didn't we swear when we took
him in we'd stamp out that dangerous nonsense?'

That evening when he got back from work, Uncle Vernon did something he'd
never done before; he visited Harry in his cupboard.

'Where's my letter?' said Harry, the moment Uncle Vernon had squeezed
through the door. 'Who's writing to me?'

'No one. it was addressed to you by mistake,' said Uncle Vernon shortly.
'I have burned it.'

'It was not a mistake,' said Harry angrily, 'it had my cupboard on it.'

'SILENCE!' yelled Uncle Vernon, and a couple of spiders fell from the
ceiling. He took a few deep breaths and then forced his face into a
smile, which looked quite painful.

'Er -- yes, Harry -- about this cupboard. Your aunt and I have been
thinking... you're really getting a bit big for it... we think it might
be nice if you moved into Dudley's second bedroom.

'Why?' said Harry.

'Don't ask questions!' snapped his uncle. 'Take this stuff upstairs,
now.'

The Dursleys' house had four bedrooms: one for Uncle Vernon and Aunt
Petunia, one for visitors (usually Uncle Vernon's sister, Merge), one
where Dudley slept, and one where Dudley kept all the toys and things
that wouldn't fit into his first bedroom. It only took Harry one trip
upstairs to move everything he owned from the cupboard to this room. He
sat down on the bed and stared around him. Nearly everything in here was
broken. The month-old video camera was lying on top of a small, working
tank Dudley had once driven over the next door neighbor's dog; in the
corner was Dudley's first-ever television set, which he'd put his foot
through when his favorite program had been canceled; there was a large
birdcage, which had once held a parrot that Dudley had swapped at school
for a real air rifle, which was up on a shelf with the end all bent
because Dudley had sat on it. Other shelves were full of books. They
were the only things in the room that looked as though they'd never been
touched.

From downstairs came the sound of Dudley bawling at his mother, I don't
want him in there... I need that room... make him get out....'

Harry sighed and stretched out on the bed. Yesterday he'd have given
anything to be up here. Today he'd rather be back in his cupboard with
that letter than up here without it.

Next morning at breakfast, everyone was rather quiet. Dudley was in
shock. He'd screamed, whacked his father with his Smelting stick, been
sick on purpose, kicked his mother, and thrown his tortoise through the
greenhouse roof, and he still didn't have his room back. Harry was
thinking about this time yesterday and bitterly wishing he'd opened the
letter in the hall. Uncle Vernon and Aunt Petunia kept looking at each
other darkly.

When the mail arrived, Uncle Vernon, who seemed to be trying to be nice
to Harry, made Dudley go and get it. They heard him banging things with
his Smelting stick all the way down the hall. Then he shouted, 'There's
another one! 'Mr. H. Potter, The Smallest Bedroom, 4 Privet Drive --''

With a strangled cry, Uncle Vernon leapt from his seat and ran down the
hall, Harry right behind him. Uncle Vernon had to wrestle Dudley to the
ground to get the letter from him, which was made difficult by the fact
that Harry had grabbed Uncle Vernon around the neck from behind. After a
minute of confused fighting, in which everyone got hit a lot by the
Smelting stick, Uncle Vernon straightened up, gasping for breath, with
Harry's letter clutched in his hand.

'Go to your cupboard -- I mean, your bedroom,' he wheezed at Harry.
'Dudley -- go -- just go.'

Harry walked round and round his new room. Someone knew he had moved out
of his cupboard and they seemed to know he hadn't received his first
letter. Surely that meant they'd try again? And this time he'd make sure
they didn't fail. He had a plan.

The repaired alarm clock rang at six o'clock the next morning. Harry
turned it off quickly and dressed silently. He mustn't wake the
Dursleys. He stole downstairs without turning on any of the lights.

He was going to wait for the postman on the corner of Privet Drive and
get the letters for number four first. His heart hammered as he crept
across the dark hall toward the front door --

Harry leapt into the air; he'd trodden on something big and squashy on
the doormat -- something alive!

Lights clicked on upstairs and to his horror Harry realized that the
big, squashy something had been his uncle's face. Uncle Vernon had been
lying at the foot of the front door in a sleeping bag, clearly making
sure that Harry didn't do exactly what he'd been trying to do. He
shouted at Harry for about half an hour and then told him to go and make
a cup of tea. Harry shuffled miserably off into the kitchen and by the
time he got back, the mail had arrived, right into Uncle Vernon's lap.
Harry could see three letters addressed in green ink.

I want --' he began, but Uncle Vernon was tearing the letters into
pieces before his eyes. Uncle Vernon didn't go to work that day. He
stayed at home and nailed up the mail slot.

'See,' he explained to Aunt Petunia through a mouthful of nails, 'if
they can't deliver them they'll just give up.'

'I'm not sure that'll work, Vernon.'

'Oh, these people's minds work in strange ways, Petunia, they're not
like you and me,' said Uncle Vernon, trying to knock in a nail with the
piece of fruitcake Aunt Petunia had just brought him.

On Friday, no less than twelve letters arrived for Harry. As they
couldn't go through the mail slot they had been pushed under the door,
slotted through the sides, and a few even forced through the small
window in the downstairs bathroom.

Uncle Vernon stayed at home again. After burning all the letters, he got
out a hammer and nails and boarded up the cracks around the front and
back doors so no one could go out. He hummed 'Tiptoe Through the Tulips'
as he worked, and jumped at small noises.

On Saturday, things began to get out of hand. Twenty-four letters to
Harry found their way into the house, rolled up and hidden inside each
of the two dozen eggs that their very confused milkman had handed Aunt
Petunia through the living room window. While Uncle Vernon made furious
telephone calls to the post office and the dairy trying to find someone
to complain to, Aunt Petunia shredded the letters in her food processor.

'Who on earth wants to talk to you this badly?' Dudley asked Harry in
amazement.

On Sunday morning, Uncle Vernon sat down at the breakfast table looking
tired and rather ill, but happy.

'No post on Sundays,' he reminded them cheerfully as he spread marmalade
on his newspapers, 'no damn letters today --'

Something came whizzing down the kitchen chimney as he spoke and caught
him sharply on the back of the head. Next moment, thirty or forty
letters came pelting out of the fireplace like bullets. The Dursleys
ducked, but Harry leapt into the air trying to catch one.

'Out! OUT!'

Uncle Vernon seized Harry around the waist and threw him into the hall.
When Aunt Petunia and Dudley had run out with their arms over their
faces, Uncle Vernon slammed the door shut. They could hear the letters
still streaming into the room, bouncing off the walls and floor.

'That does it,' said Uncle Vernon, trying to speak calmly but pulling
great tufts out of his mustache at the same time. I want you all back
here in five minutes ready to leave. We're going away. Just pack some
clothes. No arguments!'

He looked so dangerous with half his mustache missing that no one dared
argue. Ten minutes later they had wrenched their way through the
boarded-up doors and were in the car, speeding toward the highway.
Dudley was sniffling in the back seat; his father had hit him round the
head for holding them up while he tried to pack his television, VCR, and
computer in his sports bag.

They drove. And they drove. Even Aunt Petunia didn't dare ask where they
were going. Every now and then Uncle Vernon would take a sharp turn and
drive in the opposite direction for a while. 'Shake'em off... shake 'em
off,' he would mutter whenever he did this.

They didn't stop to eat or drink all day. By nightfall Dudley was
howling. He'd never had such a bad day in his life. He was hungry, he'd
missed five television programs he'd wanted to see, and he'd never gone
so long without blowing up an alien on his computer.

Uncle Vernon stopped at last outside a gloomy-looking hotel on the
outskirts of a big city. Dudley and Harry shared a room with twin beds
and damp, musty sheets. Dudley snored but Harry stayed awake, sitting on
the windowsill, staring down at the lights of passing cars and
wondering....

They ate stale cornflakes and cold tinned tomatoes on toast for
breakfast the next day. They had just finished when the owner of the
hotel came over to their table.

''Scuse me, but is one of you Mr. H. Potter? Only I got about an 'undred
of these at the front desk.'

She held up a letter so they could read the green ink address:

Mr. H. Potter

Room 17

Railview Hotel

Cokeworth

Harry made a grab for the letter but Uncle Vernon knocked his hand out
of the way. The woman stared.

'I'll take them,' said Uncle Vernon, standing up quickly and following
her from the dining room.

Wouldn't it be better just to go home, dear?' Aunt Petunia suggested
timidly, hours later, but Uncle Vernon didn't seem to hear her. Exactly
what he was looking for, none of them knew. He drove them into the
middle of a forest, got out, looked around, shook his head, got back in
the car, and off they went again. The same thing happened in the middle
of a plowed field, halfway across a suspension bridge, and at the top of
a multilevel parking garage.

'Daddy's gone mad, hasn't he?' Dudley asked Aunt Petunia dully late that
afternoon. Uncle Vernon had parked at the coast, locked them all inside
the car, and disappeared.

It started to rain. Great drops beat on the roof of the car. Dud ley
sniveled.

'It's Monday,' he told his mother. 'The Great Humberto's on tonight. I
want to stay somewhere with a television. '

Monday. This reminded Harry of something. If it was Monday -- and you
could usually count on Dudley to know the days the week, because of
television -- then tomorrow, Tuesday, was Harry's eleventh birthday. Of
course, his birthdays were never exactly fun -- last year, the Dursleys
had given him a coat hanger and a pair of Uncle Vernon's old socks.
Still, you weren't eleven every day.

Uncle Vernon was back and he was smiling. He was also carrying a long,
thin package and didn't answer Aunt Petunia when she asked what he'd
bought.

'Found the perfect place!' he said. 'Come on! Everyone out!'

It was very cold outside the car. Uncle Vernon was pointing at what
looked like a large rock way out at sea. Perched on top of the rock was
the most miserable little shack you could imagine. One thing was
certain, there was no television in there.

'Storm forecast for tonight!' said Uncle Vernon gleefully, clapping his
hands together. 'And this gentleman's kindly agreed to lend us his
boat!'

A toothless old man came ambling up to them, pointing, with a rather
wicked grin, at an old rowboat bobbing in the iron-gray water below
them.

'I've already got us some rations,' said Uncle Vernon, 'so all aboard!'

It was freezing in the boat. Icy sea spray and rain crept down their
necks and a chilly wind whipped their faces. After what seemed like
hours they reached the rock, where Uncle Vernon, slipping and sliding,
led the way to the broken-down house.

The inside was horrible; it smelled strongly of seaweed, the wind
whistled through the gaps in the wooden walls, and the fireplace was
damp and empty. There were only two rooms.

Uncle Vernon's rations turned out to be a bag of chips each and four
bananas. He tried to start a fire but the empty chip bags just smoked
and shriveled up.

'Could do with some of those letters now, eh?' he said cheerfully.

He was in a very good mood. Obviously he thought nobody stood a chance
of reaching them here in a storm to deliver mail. Harry privately
agreed, though the thought didn't cheer him up at all.

As night fell, the promised storm blew up around them. Spray from the
high waves splattered the walls of the hut and a fierce wind rattled the
filthy windows. Aunt Petunia found a few moldy blankets in the second
room and made up a bed for Dudley on the moth-eaten sofa. She and Uncle
Vernon went off to the lumpy bed next door, and Harry was left to find
the softest bit of floor he could and to curl up under the thinnest,
most ragged blanket.

The storm raged more and more ferociously as the night went on. Harry
couldn't sleep. He shivered and turned over, trying to get comfortable,
his stomach rumbling with hunger. Dudley's snores were drowned by the
low rolls of thunder that started near midnight. The lighted dial of
Dudley's watch, which was dangling over the edge of the sofa on his fat
wrist, told Harry he'd be eleven in ten minutes' time. He lay and
watched his birthday tick nearer, wondering if the Dursleys would
remember at all, wondering where the letter writer was now.

Five minutes to go. Harry heard something creak outside. He hoped the
roof wasn't going to fall in, although he might be warmer if it did.
Four minutes to go. Maybe the house in Privet Drive would be so full of
letters when they got back that he'd be able to steal one somehow.

Three minutes to go. Was that the sea, slapping hard on the rock like
that? And (two minutes to go) what was that funny crunching noise? Was
the rock crumbling into the sea?

One minute to go and he'd be eleven. Thirty seconds... twenty ... ten...
nine -- maybe he'd wake Dudley up, just to annoy him -- three... two...
one...

BOOM.

The whole shack shivered and Harry sat bolt upright, staring at the
door. Someone was outside, knocking to come in.


CHAPTER FOUR

THE KEEPER OF THE KEYS

BOOM. They knocked again. Dudley jerked awake. 'Where's the cannon?' he
said stupidly.

There was a crash behind them and Uncle Vernon came skidding into the
room. He was holding a rifle in his hands -- now they knew what had been
in the long, thin package he had brought with them.

'Who's there?' he shouted. 'I warn you -- I'm armed!'

There was a pause. Then --

SMASH!

The door was hit with such force that it swung clean off its hinges and
with a deafening crash landed flat on the floor.

A giant of a man was standing in the doorway. His face was almost
completely hidden by a long, shaggy mane of hair and a wild, tangled
beard, but you could make out his eyes, glinting like black beetles
under all the hair.

The giant squeezed his way into the hut, stooping so that his head just
brushed the ceiling. He bent down, picked up the door, and fitted it
easily back into its frame. The noise of the storm outside dropped a
little. He turned to look at them all.

'Couldn't make us a cup o' tea, could yeh? It's not been an easy
journey...'

He strode over to the sofa where Dudley sat frozen with fear.

'Budge up, yeh great lump,' said the stranger.

Dudley squeaked and ran to hide behind his mother, who was crouching,
terrified, behind Uncle Vernon.

'An' here's Harry!' said the giant.

Harry looked up into the fierce, wild, shadowy face and saw that the
beetle eyes were crinkled in a smile.

'Las' time I saw you, you was only a baby,' said the giant. 'Yeh look a
lot like yet dad, but yeh've got yet mom's eyes.'

Uncle Vernon made a funny rasping noise.

I demand that you leave at once, sit!' he said. 'You are breaking and
entering!'

'Ah, shut up, Dursley, yeh great prune,' said the giant; he reached over
the back of the sofa, jerked the gun out of Uncle Vernon's hands, bent
it into a knot as easily as if it had been made of rubber, and threw it
into a corner of the room.

Uncle Vernon made another funny noise, like a mouse being trodden on.

'Anyway -- Harry,' said the giant, turning his back on the Dursleys, 'a
very happy birthday to yeh. Got summat fer yeh here -- I mighta sat on
it at some point, but it'll taste all right.'

From an inside pocket of his black overcoat he pulled a slightly
squashed box. Harry opened it with trembling fingers. Inside was a
large, sticky chocolate cake with Happy Birthday Harry written on it in
green icing.

Harry looked up at the giant. He meant to say thank you, but the words
got lost on the way to his mouth, and what he said instead was, 'Who are
you?'

The giant chuckled.

'True, I haven't introduced meself. Rubeus Hagrid, Keeper of Keys and
Grounds at Hogwarts.'

He held out an enormous hand and shook Harry's whole arm.

'What about that tea then, eh?' he said, rubbing his hands together.
'I'd not say no ter summat stronger if yeh've got it, mind.'

His eyes fell on the empty grate with the shriveled chip bags in it and
he snorted. He bent down over the fireplace; they couldn't see what he
was doing but when he drew back a second later, there was a roaring fire
there. It filled the whole damp hut with flickering light and Harry felt
the warmth wash over him as though he'd sunk into a hot bath.

The giant sat back down on the sofa, which sagged under his weight, and
began taking all sorts of things out of the pockets of his coat: a
copper kettle, a squashy package of sausages, a poker, a teapot, several
chipped mugs, and a bottle of some amber liquid that he took a swig from
before starting to make tea. Soon the hut was full of the sound and
smell of sizzling sausage. Nobody said a thing while the giant was
working, but as he slid the first six fat, juicy, slightly burnt
sausages from the poker, Dudley fidgeted a little. Uncle Vernon said
sharply, 'Don't touch anything he gives you, Dudley.'

The giant chuckled darkly.

'Yet great puddin' of a son don' need fattenin' anymore, Dursley, don'
worry.'

He passed the sausages to Harry, who was so hungry he had never tasted
anything so wonderful, but he still couldn't take his eyes off the
giant. Finally, as nobody seemed about to explain anything, he said,
'I'm sorry, but I still don't really know who you are.'

The giant took a gulp of tea and wiped his mouth with the back of his
hand.

'Call me Hagrid,' he said, 'everyone does. An' like I told yeh, I'm
Keeper of Keys at Hogwarts -- yeh'll know all about Hogwarts, o' course.

'Er -- no,' said Harry.

Hagrid looked shocked.

'Sorry,' Harry said quickly.

'Sony?' barked Hagrid, turning to stare at the Dursleys, who shrank back
into the shadows. 'It' s them as should be sorry! I knew yeh weren't
gettin' yer letters but I never thought yeh wouldn't even know abou'
Hogwarts, fer cryin' out loud! Did yeh never wonder where yet parents
learned it all?'

'All what?' asked Harry.

'ALL WHAT?' Hagrid thundered. 'Now wait jus' one second!'

He had leapt to his feet. In his anger he seemed to fill the whole hut.
The Dursleys were cowering against the wall.

'Do you mean ter tell me,' he growled at the Dursleys, 'that this boy --
this boy! -- knows nothin' abou' -- about ANYTHING?'

Harry thought this was going a bit far. He had been to school, after
all, and his marks weren't bad.

'I know some things,' he said. 'I can, you know, do math and stuff.' But
Hagrid simply waved his hand and said, 'About our world, I mean. Your
world. My world. Yer parents' world.'

'What world?'

Hagrid looked as if he was about to explode.

'DURSLEY!' he boomed.

Uncle Vernon, who had gone very pale, whispered something that sounded
like 'Mimblewimble.' Hagrid stared wildly at Harry.

'But yeh must know about yet mom and dad,' he said. 'I mean, they're
famous. You're famous.'

'What? My -- my mom and dad weren't famous, were they?'

'Yeh don' know... yeh don' know...' Hagrid ran his fingers through his
hair, fixing Harry with a bewildered stare.

'Yeh don' know what yeh are?' he said finally.

Uncle Vernon suddenly found his voice.

'Stop!' he commanded. 'Stop right there, sit! I forbid you to tell the
boy anything!'

A braver man than Vernon Dursley would have quailed under the furious
look Hagrid now gave him; when Hagrid spoke, his every syllable trembled
with rage.

'You never told him? Never told him what was in the letter Dumbledore
left fer him? I was there! I saw Dumbledore leave it, Dursley! An'
you've kept it from him all these years?'

'Kept what from me?' said Harry eagerly.

'STOP! I FORBID YOU!' yelled Uncle Vernon in panic.

Aunt Petunia gave a gasp of horror.

'Ah, go boil yet heads, both of yeh,' said Hagrid. 'Harry -- yet a
wizard.'

There was silence inside the hut. Only the sea and the whistling wind
could be heard.

'-- a what?' gasped Harry.

'A wizard, o' course,' said Hagrid, sitting back down on the sofa, which
groaned and sank even lower, 'an' a thumpin' good'un, I'd say, once
yeh've been trained up a bit. With a mum an' dad like yours, what else
would yeh be? An' I reckon it's abou' time yeh read yer letter.'

Harry stretched out his hand at last to take the yellowish envelope,
addressed in emerald green to Mr. H. Potter, The Floor, Hut-on-the-Rock,
The Sea. He pulled out the letter and read:

HOGWARTS SCHOOL of WITCHCRAFT and WIZARDRY

Headmaster: ALBUS DUMBLEDORE

(Order of Merlin, First Class, Grand Sorc., Chf. Warlock, Supreme
Mugwump, International Confed. of Wizards)

Dear Mr. Potter,

We are pleased to inform you that you have been accepted at Hogwarts
School of Witchcraft and Wizardry. Please find enclosed a list of all
necessary books and equipment.

Term begins on September 1. We await your owl by no later than July 31.
Yours sincerely,

Minerva McGonagall,

Deputy Headmistress

Questions exploded inside Harry's head like fireworks and he couldn't
decide which to ask first. After a few minutes he stammered, 'What does
it mean, they await my owl?'

'Gallopin' Gorgons, that reminds me,' said Hagrid, clapping a hand to
his forehead with enough force to knock over a cart horse, and from yet
another pocket inside his overcoat he pulled an owl -- a real, live,
rather ruffled-looking owl -- a long quill, and a roll of parchment.
With his tongue between his teeth he scribbled a note that Harry could
read upside down:

Dear Professor Dumbledore,

Given Harry his letter.

Taking him to buy his things tomorrow.

Weather's horrible. Hope you're Well.

Hagrid

Hagrid rolled up the note, gave it to the owl, which clamped it in its
beak, went to the door, and threw the owl out into the storm. Then he
came back and sat down as though this was as normal as talking on the
telephone.

Harry realized his mouth was open and closed it quickly.

'Where was I?' said Hagrid, but at that moment, Uncle Vernon, still
ashen-faced but looking very angry, moved into the firelight.

'He's not going,' he said.

Hagrid grunted.

'I'd like ter see a great Muggle like you stop him,' he said.

'A what?' said Harry, interested.

'A Muggle,' said Hagrid, 'it's what we call nonmagic folk like thern.
An' it's your bad luck you grew up in a family o' the biggest Muggles I
ever laid eyes on.'

'We swore when we took him in we'd put a stop to that rubbish,' said
Uncle Vernon, 'swore we'd stamp it out of him! Wizard indeed!'

'You knew?' said Harry. 'You knew I'm a -- a wizard?'

'Knew!' shrieked Aunt Petunia suddenly. 'Knew! Of course we knew! How
could you not be, my dratted sister being what she was? Oh, she got a
letter just like that and disappeared off to that-that school-and came
home every vacation with her pockets full of frog spawn, turning teacups
into rats. I was the only one who saw her for what she was -- a freak!
But for my mother and father, oh no, it was Lily this and Lily that,
they were proud of having a witch in the family!'

She stopped to draw a deep breath and then went ranting on. It seemed
she had been wanting to say all this for years.

'Then she met that Potter at school and they left and got married and
had you, and of course I knew you'd be just the same, just as strange,
just as -- as -- abnormal -- and then, if you please, she went and got
herself blown up and we got landed with you!'

Harry had gone very white. As soon as he found his voice he said, 'Blown
up? You told me they died in a car crash!'

'CAR CRASH!' roared Hagrid, jumping up so angrily that the Dursleys
scuttled back to their corner. 'How could a car crash kill Lily an'
James Potter? It's an outrage! A scandal! Harry Potter not knowin' his
own story when every kid in our world knows his name!' 'But why? What
happened?' Harry asked urgently.

The anger faded from Hagrid's face. He looked suddenly anxious.

'I never expected this,' he said, in a low, worried voice. 'I had no
idea, when Dumbledore told me there might be trouble gettin' hold of
yeh, how much yeh didn't know. Ah, Harry, I don' know if I'm the right
person ter tell yeh -- but someone 3 s gotta -- yeh can't go off ter
Hogwarts not knowin'.'

He threw a dirty look at the Dursleys.

'Well, it's best yeh know as much as I can tell yeh -- mind, I can't
tell yeh everythin', it's a great myst'ry, parts of it....'

He sat down, stared into the fire for a few seconds, and then said, 'It
begins, I suppose, with -- with a person called -- but it's incredible
yeh don't know his name, everyone in our world knows --'

'Who? '

'Well -- I don' like sayin' the name if I can help it. No one does.'

'Why not?'

'Gulpin' gargoyles, Harry, people are still scared. Blimey, this is
difficult. See, there was this wizard who went... bad. As bad as you
could go. Worse. Worse than worse. His name was...'

Hagrid gulped, but no words came out.

'Could you write it down?' Harry suggested.

'Nah -can't spell it. All right -- Voldemort. ' Hagrid shuddered. 'Don'
make me say it again. Anyway, this -- this wizard, about twenty years
ago now, started lookin' fer followers. Got 'em, too -- some were
afraid, some just wanted a bit o' his power, 'cause he was gettin'
himself power, all right. Dark days, Harry. Didn't know who ter trust,
didn't dare get friendly with strange wizards or witches... terrible
things happened. He was takin' over. 'Course, some stood up to him --
an' he killed 'em. Horribly. One o' the only safe places left was
Hogwarts. Reckon Dumbledore's the only one You-Know-Who was afraid of.
Didn't dare try takin' the school, not jus' then, anyway.

'Now, yer mum an' dad were as good a witch an' wizard as I ever knew.
Head boy an' girl at Hogwarts in their day! Suppose the myst'ry is why
You-Know-Who never tried to get 'em on his side before... probably knew
they were too close ter Dumbledore ter want anythin' ter do with the
Dark Side.

'Maybe he thought he could persuade 'em... maybe he just wanted 'em
outta the way. All anyone knows is, he turned up in the village where
you was all living, on Halloween ten years ago. You was just a year old.
He came ter yer house an' -- an' --'

Hagrid suddenly pulled out a very dirty, spotted handkerchief and blew
his nose with a sound like a foghorn.

'Sorry,' he said. 'But it's that sad -- knew yer mum an' dad, an' nicer
people yeh couldn't find -- anyway...'

'You-Know-Who killed 'em. An' then -- an' this is the real myst'ry of
the thing -- he tried to kill you, too. Wanted ter make a clean job of
it, I suppose, or maybe he just liked killin' by then. But he couldn't
do it. Never wondered how you got that mark on yer forehead? That was no
ordinary cut. That's what yeh get when a Powerful, evil curse touches
yeh -- took care of yer mum an' dad an' yer house, even -- but it didn't
work on you, an' that's why yer famous, Harry. No one ever lived after
he decided ter kill 'em, no one except you, an' he'd killed some o' the
best witches an' wizards of the age -- the McKinnons, the Bones, the
Prewetts -- an' you was only a baby, an' you lived.'

Something very painful was going on in Harry's mind. As Hagrid's story
came to a close, he saw again the blinding flash of green light, more
clearly than he had ever remembered it before -- and he remembered
something else, for the first time in his life: a high, cold, cruel
laugh.

Hagrid was watching him sadly.

'Took yeh from the ruined house myself, on Dumbledore's orders. Brought
yeh ter this lot...'

'Load of old tosh,' said Uncle Vernon. Harry jumped; he had almost
forgotten that the Dursleys were there. Uncle Vernon certainly seemed to
have got back his courage. He was glaring at Hagrid and his fists were
clenched.

'Now, you listen here, boy,' he snarled, 'I accept there's something
strange about you, probably nothing a good beating wouldn't have cured
-- and as for all this about your parents, well, they were weirdos, no
denying it, and the world's better off without them in my opinion --
asked for all they got, getting mixed up with these wizarding types --
just what I expected, always knew they'd come to a sticky end --'

But at that moment, Hagrid leapt from the sofa and drew a battered pink
umbrella from inside his coat. Pointing this at Uncle Vernon like a
sword, he said, 'I'm warning you, Dursley -I'm warning you -- one more
word... '

In danger of being speared on the end of an umbrella by a bearded giant,
Uncle Vernon's courage failed again; he flattened himself against the
wall and fell silent.

'That's better,' said Hagrid, breathing heavily and sitting back down on
the sofa, which this time sagged right down to the floor.

Harry, meanwhile, still had questions to ask, hundreds of them.

'But what happened to Vol--, sorry -- I mean, You-Know-Who?'

'Good question, Harry. Disappeared. Vanished. Same night he tried ter
kill you. Makes yeh even more famous. That's the biggest myst'ry, see...
he was gettin' more an' more powerful -- why'd he go?

'Some say he died. Codswallop, in my opinion. Dunno if he had enough
human left in him to die. Some say he's still out there, bidin' his
time, like, but I don' believe it. People who was on his side came back
ter ours. Some of 'em came outta kinda trances. Don~ reckon they
could've done if he was comin' back.

'Most of us reckon he's still out there somewhere but lost his powers.
Too weak to carry on. 'Cause somethin' about you finished him, Harry.
There was somethin' goin' on that night he hadn't counted on -- I dunno
what it was, no one does -- but somethin' about you stumped him, all
right.'

Hagrid looked at Harry with warmth and respect blazing in his eyes, but
Harry, instead of feeling pleased and proud, felt quite sure there had
been a horrible mistake. A wizard? Him? How could he possibly be? He'd
spent his life being clouted by Dudley, and bullied by Aunt Petunia and
Uncle Vernon; if he was really a wizard, why hadn't they been turned
into warty toads every time they'd tried to lock him in his cupboard? If
he'd once defeated the greatest sorcerer in the world, how come Dudley
had always been able to kick him around like a football?

'Hagrid,' he said quietly, 'I think you must have made a mistake. I
don't think I can be a wizard.'

To his surprise, Hagrid chuckled.

'Not a wizard, eh? Never made things happen when you was scared or
angry?'

Harry looked into the fire. Now he came to think about it... every odd
thing that had ever made his aunt and uncle furious with him had
happened when he, Harry, had been upset or angry... chased by Dudley's
gang, he had somehow found himself out of their reach... dreading going
to school with that ridiculous haircut, he'd managed to make it grow
back... and the very last time Dudley had hit him, hadn't he got his
revenge, without even realizing he was doing it? Hadn't he set a boa
constrictor on him?

Harry looked back at Hagrid, smiling, and saw that Hagrid was positively
beaming at him.

'See?' said Hagrid. 'Harry Potter, not a wizard -- you wait, you'll be
right famous at Hogwarts.'

But Uncle Vernon wasn't going to give in without a fight.

'Haven't I told you he's not going?' he hissed. 'He's going to Stonewall
High and he'll be grateful for it. I've read those letters and he needs
all sorts of rubbish -- spell books and wands and --'

'If he wants ter go, a great Muggle like you won't stop him,' growled
Hagrid. 'Stop Lily an' James Potter' s son goin' ter Hogwarts! Yer mad.
His name's been down ever since he was born. He's off ter the finest
school of witchcraft and wizardry in the world. Seven years there and he
won't know himself. He'll be with youngsters of his own sort, fer a
change, an' he'll be under the greatest headmaster Hogwarts ever had
Albus Dumbled--'

'I AM NOT PAYING FOR SOME CRACKPOT OLD FOOL To TEACH HIM MAGIC TRICKS!'
yelled Uncle Vernon.

But he had finally gone too far. Hagrid seized his umbrella and whirled
it over his head, 'NEVER,' he thundered, '- INSULT- ALBUS- DUMBLEDORE-
IN- FRONT- OF- ME!'

He brought the umbrella swishing down through the air to point at Dudley
-- there was a flash of violet light, a sound like a firecracker, a
sharp squeal, and the next second, Dudley was dancing on the spot with
his hands clasped over his fat bottom, howling in pain. When he turned
his back on them, Harry saw a curly pig's tail poking through a hole in
his trousers.

Uncle Vernon roared. Pulling Aunt Petunia and Dudley into the other
room, he cast one last terrified look at Hagrid and slammed the door
behind them.

Hagrid looked down at his umbrella and stroked his beard.

'Shouldn'ta lost me temper,' he said ruefully, 'but it didn't work
anyway. Meant ter turn him into a pig, but I suppose he was so much like
a pig anyway there wasn't much left ter do.'

He cast a sideways look at Harry under his bushy eyebrows.

'Be grateful if yeh didn't mention that ter anyone at Hogwarts,' he
said. 'I'm -- er -- not supposed ter do magic, strictly speakin'. I was
allowed ter do a bit ter follow yeh an' get yer letters to yeh an' stuff
-- one o' the reasons I was so keen ter take on the job

'Why aren't you supposed to do magic?' asked Harry.

'Oh, well -- I was at Hogwarts meself but I -- er -- got expelled, ter
tell yeh the truth. In me third year. They snapped me wand in half an'
everything. But Dumbledore let me stay on as gamekeeper. Great man,
Dumbledore.' 'Why were you expelled?'

'It's gettin' late and we've got lots ter do tomorrow,' said Hagrid
loudly. 'Gotta get up ter town, get all yer books an' that.'

He took off his thick black coat and threw it to Harry.

'You can kip under that,' he said. 'Don' mind if it wriggles a bit, I
think I still got a couple o' dormice in one o' the pockets.'


CHAPTER FIVE

DIAGON ALLEY

Harry woke early the next morning. Although he could tell it was
daylight, he kept his eyes shut tight.

'It was a dream, he told himself firmly. 'I dreamed a giant called
Hagrid came to tell me I was going to a school for wizards. When I open
my eyes I'll be at home in my cupboard.'

There was suddenly a loud tapping noise.

And there's Aunt Petunia knocking on the door, Harry thought, his heart
sinking. But he still didn't open his eyes. It had been such a good
dream.

Tap. Tap. Tap.

'All right,' Harry mumbled, 'I'm getting up.'

He sat up and Hagrid's heavy coat fell off him. The hut was full of
sunlight, the storm was over, Hagrid himself was asleep on the collapsed
sofa, and there was an owl rapping its claw on the window, a newspaper
held in its beak.

Harry scrambled to his feet, so happy he felt as though a large balloon
was swelling inside him. He went straight to the window and jerked it
open. The owl swooped in and dropped the newspaper on top of Hagrid, who
didn't wake up. The owl then fluttered onto the floor and began to
attack Hagrid's coat.

'Don't do that.'

Harry tried to wave the owl out of the way, but it snapped its beak
fiercely at him and carried on savaging the coat.

'Hagrid!' said Harry loudly. 'There's an owl

'Pay him,' Hagrid grunted into the sofa.

'What?'

'He wants payin' fer deliverin' the paper. Look in the pockets.'
Hagrid's coat seemed to be made of nothing but pockets -- bunches of
keys, slug pellets, balls of string, peppermint humbugs, teabags...
finally, Harry pulled out a handful of strange-looking coins.

'Give him five Knuts,' said Hagrid sleepily.

'Knuts?'

'The little bronze ones.'

Harry counted out five little bronze coins, and the owl held out his leg
so Harry could put the money into a small leather pouch tied to it. Then
he flew off through the open window.

Hagrid yawned loudly, sat up, and stretched.

'Best be Off, Harry, lots ter do today, gotta get up ter London an' buy
all yer stuff fer school.'

Harry was turning over the wizard coins and looking at them. He had just
thought of something that made him feel as though the happy balloon
inside him had got a puncture.

'Um -- Hagrid?'

'Mm?' said Hagrid, who was pulling on his huge boots.

'I haven't got any money -- and you heard Uncle Vernon last night ... he
won't pay for me to go and learn magic.'

'Don't worry about that,' said Hagrid, standing up and scratching his
head. 'D'yeh think yer parents didn't leave yeh anything?'

'But if their house was destroyed --'

'They didn' keep their gold in the house, boy! Nah, first stop fer us is
Gringotts. Wizards' bank. Have a sausage, they're not bad cold -- an' I
wouldn' say no the a bit o' yer birthday cake, neither.'

'Wizards have banks?'

'Just the one. Gringotts. Run by goblins.'

Harry dropped the bit of sausage he was holding.

'Goblins?'

'Yeah -- so yeh'd be mad ter try an' rob it, I'll tell yeh that. Never
mess with goblins, Harry. Gringotts is the safest place in the world fer
anything yeh want ter keep safe -- 'cept maybe Hogwarts. As a matter o'
fact, I gotta visit Gringotts anyway. Fer Dumbledore. Hogwarts
business.' Hagrid drew himself up proudly. 'He usually gets me ter do
important stuff fer him. Fetchin' you gettin' things from Gringotts --
knows he can trust me, see.

'Got everythin'? Come on, then.'

Harry followed Hagrid out onto the rock. The sky was quite clear now and
the sea gleamed in the sunlight. The boat Uncle Vernon had hired was
still there, with a lot of water in the bottom after the storm.

'How did you get here?' Harry asked, looking around for another boat.
'Flew,' said Hagrid.

'Flew?'

'Yeah -- but we'll go back in this. Not s'pposed ter use magic now I've
got yeh.'

They settled down in the boat, Harry still staring at Hagrid, trying to
imagine him flying.

'Seems a shame ter row, though,' said Hagrid, giving Harry another of
his sideways looks. 'If I was ter -- er -- speed things up a bit, would
yeh mind not mentionin' it at Hogwarts?'

'Of course not,' said Harry, eager to see more magic. Hagrid pulled out
the pink umbrella again, tapped it twice on the side of the boat, and
they sped off toward land.

'Why would you be mad to try and rob Gringotts?' Harry asked.

'Spells -- enchantments,' said Hagrid, unfolding his newspaper as he
spoke. 'They say there's dragons guardin' the highsecurity vaults. And
then yeh gotta find yer way -- Gringotts is hundreds of miles under
London, see. Deep under the Underground. Yeh'd die of hunger tryin' ter
get out, even if yeh did manage ter get yer hands on summat.'

Harry sat and thought about this while Hagrid read his newspaper, the
Daily Prophet. Harry had learned from Uncle Vernon that people liked to
be left alone while they did this, but it was very difficult, he'd never
had so many questions in his life.

'Ministry o' Magic messin' things up as usual,' Hagrid muttered, turning
the page.

'There's a Ministry of Magic?' Harry asked, before he could stop
himself.

''Course,' said Hagrid. 'They wanted Dumbledore fer Minister, 0 '
course, but he'd never leave Hogwarts, so old Cornelius Fudge got the
job. Bungler if ever there was one. So he pelts Dumbledore with owls
every morning, askin' fer advice.'

'But what does a Ministry of Magic do?'

'Well, their main job is to keep it from the Muggles that there's still
witches an' wizards up an' down the country.'

'Why?'

'Why? Blimey, Harry, everyone'd be wantin' magic solutions to their
problems. Nah, we're best left alone.'

At this moment the boat bumped gently into the harbor wall. Hagrid
folded up his newspaper, and they clambered up the stone steps onto the
street.
"""


def main():
    args = get_args()
    depth_percent = args.depth_percent

    assert args.ckpt_path.exists(), f"Checkpoint path {args.ckpt_path} does not exist"

    config = get_config_from_file((args.ckpt_path / "config.yaml").as_posix())
    model_config = config.model.model_config
    tokenizer_path = config.tokenizer.tokenizer_name_or_path

    parallel_config = ParallelismArgs(
        dp=args.dp or config.parallelism.dp,
        pp=args.pp or config.parallelism.pp,
        tp=args.tp or config.parallelism.tp,
        pp_engine=OneForwardOneBackwardPipelineEngine(),
        tp_mode=TensorParallelLinearMode.REDUCE_SCATTER,
        tp_linear_async_communication=True,
    )

    # Initialise all process groups
    parallel_context = ParallelContext(
        data_parallel_size=parallel_config.dp,
        pipeline_parallel_size=parallel_config.pp,
        tensor_parallel_size=parallel_config.tp,
    )

    # Set log levels
    logging_config = LoggingArgs(
        log_level="info",
        log_level_replica="info",
    )

    # Set log levels
    set_ranks_logging_level(parallel_context=parallel_context, logging_config=logging_config)

    log_rank(f"model_config: {model_config}", logger=logger, level=logging.INFO, rank=0)
    log_rank(f"tokenizer_path: {tokenizer_path}", logger=logger, level=logging.INFO, rank=0)

    dtype = torch.bfloat16

    # Set random states
    set_random_seed(42)

    model_config_cls = model_config.__class__.__name__
    if model_config_cls not in CONFIG_TO_MODEL_CLASS:
        raise ValueError(
            f"Unsupported model config {model_config_cls}. Only {CONFIG_TO_MODEL_CLASS.keys()} are supported"
        )

    # Get synchronized random states
    if parallel_config.tp_mode is TensorParallelLinearMode.ALL_REDUCE:
        random_states = RandomStates(
            {"tp_synced": get_synced_random_state(random_state=get_current_random_state(), pg=parallel_context.tp_pg)}
        )
    else:
        # We don't need to sync across TP when using sequence parallel (REDUCE_SCATTER)
        random_states = RandomStates({})

    model = build_model(
        model_builder=lambda: CONFIG_TO_MODEL_CLASS[model_config_cls](
            config=model_config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
            random_states=random_states,
        ),
        dtype=dtype,
        parallel_context=parallel_context,
    )

    # Mark some parameters as tied
    # TODO @nouamane: this is only needed for training, can we just mark params as NanotronParameter instead?
    mark_tied_parameters(model=model, parallel_context=parallel_context, parallel_config=parallel_config)

    # Sanity check model
    sanity_check(root_module=model)

    # Load checkpoint
    checkpoint_path = args.ckpt_path
    log_rank(
        f"Loading checkpoint from {checkpoint_path}:",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )
    load_weights(model=model, parallel_context=parallel_context, root_folder=checkpoint_path)

    model.eval()
    if AutoTokenizer is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # tokenizer.pad_token_id = tokenizer.eos_token_id
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            elif getattr(model.config, "pad_token_id", None) is not None:
                tokenizer.pad_token_id = int(model.config.pad_token_id)
            elif getattr(model.config, "eos_token_id", None) is not None:
                tokenizer.pad_token_id = int(model.config.eos_token_id)
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"  # TODO @nouamane: do we want this?

        import wandb

        if dist.get_rank() == 0:
            wandb.init(
                project="debug_infini_attention",
                name="debug_infini_attention",
            )

        # dummy_inputs = [
        #     # "Passage: Daniel went back to the garden. Mary travelled to the kitchen. Sandra journeyed to the kitchen. Sandra went to the hallway. John went to the bedroom. Mary went back to the garden. Where is Mary?\nAnswer:",
        #     # "Passage: Daniel went back to the garden. Mary travelled to the kitchen. Sandra journeyed to the kitchen. Sandra went to the hallway. John went to the bedroom. Mary went back to the garden. Where is Mary?\nAnswer:",
        #     # "Passage: Daniel went back to the garden. Mary travelled to the kitchen. Sandra journeyed to the kitchen. Sandra went to the hallway. John went to the bedroom. Mary went back to the garden. Where is Mary?\nAnswer:",
        #     # "def fib(n)",
        #     # "This film was probably inspired by Godzilla",
        #     # END_PASSKEY,
        #     PASSKEY_NINETY_PERCENT,
        #     # FINETUNE
        # ]

        # generate(args, model, tokenizer, [HARRY_POTTER], parallel_context)

        # dataset = load_dataset("nanotron/simple_needle_in_a_hay_stack", split="train")
        # df = load_dataset("nanotron/simple_needle_in_a_hay_stack", split="train")
        # from datasets import load_dataset

        # dataset = load_dataset("lvwerra/needle-llama3-16x512", split="train")
        # df = load_dataset("lvwerra/needle-llama3-16x512", split="train")

        # NOTE: filter out only samples with context_length is 32768
        dataset = dataset.filter(lambda x: x["context_length"] == 32768 and x["depth_percent"] == depth_percent)
        df = df.filter(lambda x: x["context_length"] == 32768 and x["depth_percent"] == depth_percent)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        df.set_format("pandas")
        df = df[:]

        responses = []
        from tqdm import tqdm

        for batch in tqdm(dataloader):
            print("--------------------------------------------------")
            print(f"target answer: {batch['answer']}")
            texts = batch["prompt"]
            from nanotron import constants

            constants.NEEDLE = batch["answer"].item()

            responses.append(generate(args, model, tokenizer, texts, parallel_context))

        # NOTE: now flatten the responses
        responses = [response for sublist in responses for response in sublist]
        df["response"] = responses
        # df["match"] = df.apply(lambda x: int(str(x["needle"]) in x["response"]), axis=1)

        # NOTE: move anything from gpu to cpu in df

        # nOTE: now save df
        # df.to_pickle(f'needle_finetune_format_dataset_but_for_evals_{context_length}_ctx_and_{depth_percent}_depth.pkl')

    dist.barrier()


if __name__ == "__main__":
    main()
