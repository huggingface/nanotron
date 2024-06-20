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


TEXT = """The United States Army (USA) is the land service branch of the United States Armed Forces. It is one of the eight U.S. uniformed services, and is designated as the Army of the United States in the U.S. Constitution.[14] The Army is the oldest branch of the U.S. military and the most senior in order of precedence.[15] It has its roots in the Continental Army, which was formed on 14 June 1775 to fight against the British for independence during the American Revolutionary War (1775–1783).[16] After the Revolutionary War, the Congress of the Confederation created the United States Army on 3 June 1784 to replace the disbanded Continental Army.[17][18] The United States Army considers itself a continuation of the Continental Army, and thus considers its institutional inception to be the origin of that armed force in 1775.[16]

The U.S. Army is a uniformed service of the United States and is part of the Department of the Army, which is one of the three military departments of the Department of Defense. The U.S. Army is headed by a civilian senior appointed civil servant, the secretary of the Army (SECARMY), and by a chief military officer, the chief of staff of the Army (CSA) who is also a member of the Joint Chiefs of Staff. It is the largest military branch, and in the fiscal year 2022, the projected end strength for the Regular Army (USA) was 480,893 soldiers; the Army National Guard (ARNG) had 336,129 soldiers and the U.S. Army Reserve (USAR) had 188,703 soldiers; the combined-component strength of the U.S. Army was 1,005,725 soldiers.[19] As a branch of the armed forces, the mission of the U.S. Army is 'to fight and win our Nation's wars, by providing prompt, sustained land dominance, across the full range of military operations and the spectrum of conflict, in support of combatant commanders'.[20] The branch participates in conflicts worldwide and is the major ground-based offensive and defensive force of the United States of America.‌

Mission
The United States Army serves as the land-based branch of the U.S. Armed Forces. Section 7062 of Title 10, U.S. Code defines the purpose of the army as:[21][22]

Preserving the peace and security and providing for the defense of the United States, the Commonwealths and possessions, and any areas occupied by the United States
Supporting the national policies
Implementing the national objectives
Overcoming any nations responsible for aggressive acts that imperil the peace and security of the United States
In 2018, the Army Strategy 2018 articulated an eight-point addendum to the Army Vision for 2028.[23] While the Army Mission remains constant, the Army Strategy builds upon the Army's Brigade Modernization by adding focus to corps and division-level echelons.[23] The Army Futures Command oversees reforms geared toward conventional warfare. The Army's current reorganization plan is due to be completed by 2028.[23]

The Army's five core competencies are prompt and sustained land combat, combined arms operations (to include combined arms maneuver and wide–area security, armored and mechanized operations and airborne and air assault operations), special operations forces, to set and sustain the theater for the joint force, and to integrate national, multinational, and joint power on land.[24]

History
Main article: History of the United States Army
Origins
The Continental Army was created on 14 June 1775 by the Second Continental Congress[25] as a unified army for the colonies to fight Great Britain, with George Washington appointed as its commander.[16][26][27][28] The army was initially led by men who had served in the British Army or colonial militias and who brought much of British military heritage with them. As the Revolutionary War progressed, French aid, resources, and military thinking helped shape the new army. A number of European soldiers came on their own to help, such as Friedrich Wilhelm von Steuben, who taught Prussian Army tactics and organizational skills.


The storming of Redoubt No. 10 in the Siege of Yorktown during the American Revolutionary War prompted Great Britain's government to begin negotiations, resulting in the Treaty of Paris and Great Britain's recognition of the United States as an independent state.
The Army fought numerous pitched battles, and sometimes used Fabian strategy and hit-and-run tactics in the South in 1780 and 1781; under Major General Nathanael Greene, it hit where the British were weakest to wear down their forces. Washington led victories against the British at Trenton and Princeton, but lost a series of battles in the New York and New Jersey campaign in 1776 and the Philadelphia campaign in 1777. With a decisive victory at Yorktown and the help of the French, the Continental Army prevailed against the British.

After the war, the Continental Army was quickly given land certificates and disbanded in a reflection of the republican distrust of standing armies. State militias became the new nation's sole ground army, except a regiment to guard the Western Frontier and one battery of artillery guarding West Point's arsenal. However, because of continuing conflict with Native Americans, it was soon considered necessary to field a trained standing army. The Regular Army was at first very small and after General St. Clair's defeat at the Battle of the Wabash,[29] where more than 800 soldiers were killed, the Regular Army was reorganized as the Legion of the United States, established in 1791 and renamed the United States Army in 1796.

In 1798, during the Quasi-War with France, the U.S. Congress established a three-year 'Provisional Army' of 10,000 men, consisting of twelve regiments of infantry and six troops of light dragoons. In March 1799, Congress created an 'Eventual Army' of 30,000 men, including three regiments of cavalry. Both 'armies' existed only on paper, but equipment for 3,000 men and horses was procured and stored.[30]

19th century
War of 1812 and Indian Wars
Further information: War of 1812 and Army on the Frontier

General Andrew Jackson standing on the parapet of his makeshift defenses as his troops repulse attacking Highlanders during the defense of New Orleans, the final major and most one-sided battle of the War of 1812, mainly fought by militia and volunteers.
The War of 1812, the second and last war between the United States and Great Britain, had mixed results. The U.S. Army did not conquer Canada but it did destroy Native American resistance to expansion in the Old Northwest and it validated its independence by stopping two major British invasions in 1814 and 1815. After taking control of Lake Erie in 1813, the U.S. Army seized parts of western Upper Canada, burned York and defeated Tecumseh, which caused his Western Confederacy to collapse. Following U.S. victories in the Canadian province of Upper Canada, British troops who had dubbed the U.S. Army 'Regulars, by God!', were able to capture and burn Washington, which was defended by militia, in 1814. The regular army, however, proved they were professional and capable of defeating the British army during the invasions of Plattsburgh and Baltimore, prompting British agreement on the previously rejected terms of a status quo antebellum.[dubious – discuss] Two weeks after a treaty was signed (but not ratified), Andrew Jackson defeated the British in the Battle of New Orleans and siege of Fort St. Philip with an army dominated by militia and volunteers, and became a national hero. U.S. troops and sailors captured HMS Cyane, Levant and Penguin in the final engagements of the war. Per the treaty, both sides (the United States and Great Britain) returned to the geographical status quo. Both navies kept the warships they had seized during the conflict.

The army's major campaign against the Indians was fought in Florida against Seminoles. It took long wars (1818–1858) to finally defeat the Seminoles and move them to Oklahoma. The usual strategy in Indian wars was to seize control of the Indians' winter food supply, but that was no use in Florida where there was no winter. The second strategy was to form alliances with other Indian tribes, but that too was useless because the Seminoles had destroyed all the other Indians when they entered Florida in the late eighteenth century.[31]

The U.S. Army fought and won the Mexican–American War (1846–1848), which was a defining event for both countries.[32] The U.S. victory resulted in acquisition of territory that eventually became all or parts of the states of California, Nevada, Utah, Colorado, Arizona, Wyoming and New Mexico.

American Civil War
Further information: Union Army

The Battle of Gettysburg, the turning point of the American Civil War
The American Civil War was the costliest war for the U.S. in terms of casualties. After most slave states, located in the southern U.S., formed the Confederate States, the Confederate States Army, led by former U.S. Army officers, mobilized a large fraction of Southern white manpower. Forces of the United States (the 'Union' or 'the North') formed the Union Army, consisting of a small body of regular army units and a large body of volunteer units raised from every state, north and south, except South Carolina.[33]

For the first two years, Confederate forces did well in set battles but lost control of the border states.[34] The Confederates had the advantage of defending a large territory in an area where disease caused twice as many deaths as combat. The Union pursued a strategy of seizing the coastline, blockading the ports, and taking control of the river systems. By 1863, the Confederacy was being strangled. Its eastern armies fought well, but the western armies were defeated one after another until the Union forces captured New Orleans in 1862 along with the Tennessee River. In the Vicksburg Campaign of 1862–1863, General Ulysses Grant seized the Mississippi River and cut off the Southwest. Grant took command of Union forces in 1864 and after a series of battles with very heavy casualties, he had General Robert E. Lee under siege in Richmond as General William T. Sherman captured Atlanta and marched through Georgia and the Carolinas. The Confederate capital was abandoned in April 1865 and Lee subsequently surrendered his army at Appomattox Court House. All other Confederate armies surrendered within a few months.

The war remains the deadliest conflict in U.S. history, resulting in the deaths of 620,000 men on both sides. Based on 1860 census figures, 8% of all white males aged 13 to 43 died in the war, including 6.4% in the North and 18% in the South.[35]

Later 19th century

Army soldiers in 1890
Following the Civil War, the U.S. Army had the mission of containing western tribes of Native Americans on the Indian reservations. They set up many forts, and engaged in the last of the American Indian Wars. U.S. Army troops also occupied several Southern states during the Reconstruction Era to protect freedmen.

The key battles of the Spanish–American War of 1898 were fought by the Navy. Using mostly new volunteers, the U.S. forces defeated Spain in land campaigns in Cuba and played the central role in the Philippine–American War.

20th century
Starting in 1910, the army began acquiring fixed-wing aircraft.[36] In 1910, during the Mexican Revolution, the army was deployed to U.S. towns near the border to ensure the safety of lives and property. In 1916, Pancho Villa, a major rebel leader, attacked Columbus, New Mexico, prompting a U.S. intervention in Mexico until 7 February 1917. They fought the rebels and the Mexican federal troops until 1918.

World Wars
For a list of campaigns see List of United States Army campaigns during World War II

U.S. Army troops assaulting a German bunker in France, c. 1918
The United States joined World War I as an 'Associated Power' in 1917 on the side of Britain, France, Russia, Italy and the other Allies. U.S. troops were sent to the Western Front and were involved in the last offensives that ended the war. With the armistice in November 1918, the army once again decreased its forces.

In 1939, estimates of the Army's strength ranged between 174,000 and 200,000 soldiers, smaller than that of Portugal's, which ranked it 17th or 19th in the world in size. General George C. Marshall became Army chief of staff in September 1939 and set about expanding and modernizing the Army in preparation for war.[37][38]


U.S. soldiers hunting for Japanese infiltrators during the Bougainville Campaign
The United States joined World War II in December 1941 after the Japanese attack on Pearl Harbor. Some 11 million Americans were to serve in various Army operations.[39][40] On the European front, U.S. Army troops formed a significant portion of the forces that landed in French North Africa and took Tunisia and then moved on to Sicily and later fought in Italy. In the June 1944 landings in northern France and in the subsequent liberation of Europe and defeat of Nazi Germany, millions of U.S. Army troops played a central role. In 1947, the number of soldiers in the US Army had decreased from eight million in 1945 to 684,000 soldiers and the total number of active divisions had dropped from 89 to 12. The leaders of the Army saw this demobilization as a success.[41] In the Pacific War, U.S. Army soldiers participated alongside the United States Marine Corps in capturing the Pacific Islands from Japanese control. Following the Axis surrenders in May (Germany) and August (Japan) of 1945, army troops were deployed to Japan and Germany to occupy the two defeated nations. Two years after World War II, the Army Air Forces separated from the army to become the United States Air Force in September 1947. In 1948, the army was desegregated by order 9981 of President Harry S. Truman.

Cold War
1945–1960

U.S. Army soldiers observing an atomic bomb test of Operation Buster-Jangle at the Nevada Test Site during the Korean War
The end of World War II set the stage for the East–West confrontation known as the Cold War. With the outbreak of the Korean War, concerns over the defense of Western Europe rose. Two corps, V and VII, were reactivated under Seventh United States Army in 1950 and U.S. strength in Europe rose from one division to four. Hundreds of thousands of U.S. troops remained stationed in West Germany, with others in Belgium, the Netherlands and the United Kingdom, until the 1990s in anticipation of a possible Soviet attack.[42]: minute 9:00–10:00


US tanks and Soviet tanks at Checkpoint Charlie, 1961
During the Cold War, U.S. troops and their allies fought communist forces in Korea and Vietnam. The Korean War began in June 1950, when the Soviets walked out of a UN Security Council meeting, removing their possible veto. Under a United Nations umbrella, hundreds of thousands of U.S. troops fought to prevent the takeover of South Korea by North Korea and later to invade the northern nation. After repeated advances and retreats by both sides and the Chinese People's Volunteer Army's entry into the war, the Korean Armistice Agreement returned the peninsula to the status quo in July 1953.

1960–1970
The Vietnam War is often regarded as a low point for the U.S. Army due to the use of drafted personnel, the unpopularity of the war with the U.S. public and frustrating restrictions placed on the military by U.S. political leaders. While U.S. forces had been stationed in South Vietnam since 1959, in intelligence and advising/training roles, they were not deployed in large numbers until 1965, after the Gulf of Tonkin Incident. U.S. forces effectively established and maintained control of the 'traditional' battlefield, but they struggled to counter the guerrilla hit and run tactics of the communist Viet Cong and the People's Army Of Vietnam (NVA).[43][44]


A U.S. Army infantry patrol moving up to assault the last North Vietnamese Army position at Dak To, South Vietnam during Operation Hawthorne
During the 1960s, the Department of Defense continued to scrutinize the reserve forces and to question the number of divisions and brigades as well as the redundancy of maintaining two reserve components, the Army National Guard and the Army Reserve.[45] In 1967, Secretary of Defense Robert McNamara decided that 15 combat divisions in the Army National Guard were unnecessary and cut the number to eight divisions (one mechanized infantry, two armored, and five infantry), but increased the number of brigades from seven to 18 (one airborne, one armored, two mechanized infantry and 14 infantry). The loss of the divisions did not sit well with the states. Their objections included the inadequate maneuver element mix for those that remained and the end to the practice of rotating divisional commands among the states that supported them. Under the proposal, the remaining division commanders were to reside in the state of the division base. However, no reduction in total Army National Guard strength was to take place, which convinced the governors to accept the plan. The states reorganized their forces accordingly between 1 December 1967 and 1 May 1968.

1970–1990

U.S. Army soldiers preparing to take La Comandancia in the El Chorrillo neighborhood of Panama City during Operation Just Cause
The Total Force Policy was adopted by Chief of Staff of the Army General Creighton Abrams in the aftermath of the Vietnam War and involved treating the three components of the army – the Regular Army, the Army National Guard and the Army Reserve as a single force.[46] General Abrams' intertwining of the three components of the army effectively made extended operations impossible without the involvement of both the Army National Guard and Army Reserve in a predominantly combat support role.[47] The army converted to an all-volunteer force with greater emphasis on training to specific performance standards driven by the reforms of General William E. DePuy, the first commander of United States Army Training and Doctrine Command. Following the Camp David Accords that was signed by Egypt, Israel that was brokered by president Jimmy Carter in 1978, as part of the agreement, both the United States and Egypt agreed that there would be a joint military training led by both countries that would usually take place every 2 years, that exercise is known as Exercise Bright Star.

The 1980s was mostly a decade of reorganization. The Goldwater-Nichols Act of 1986 created unified combatant commands bringing the army together with the other four military services under unified, geographically organized command structures. The army also played a role in the invasions of Grenada in 1983 (Operation Urgent Fury) and Panama in 1989 (Operation Just Cause).

By 1989 Germany was nearing reunification and the Cold War was coming to a close. Army leadership reacted by starting to plan for a reduction in strength. By November 1989 Pentagon briefers were laying out plans to reduce army end strength by 23%, from 750,000 to 580,000.[48] A number of incentives such as early retirement were used.

1990s

M1 Abrams tanks moving out before the Battle of Al Busayyah during the Gulf War
In 1990, Iraq invaded its smaller neighbor, Kuwait, and U.S. land forces quickly deployed to assure the protection of Saudi Arabia. In January 1991 Operation Desert Storm commenced, a U.S.-led coalition which deployed over 500,000 troops, the bulk of them from U.S. Army formations, to drive out Iraqi forces. The campaign ended in total victory, as Western coalition forces routed the Iraqi Army. Some of the largest tank battles in history were fought during the Gulf war. The Battle of Medina Ridge, Battle of Norfolk and the Battle of 73 Easting were tank battles of historical significance.[49][50][51]


Iraqi tanks destroyed by Task Force 1-41 Infantry during the Gulf War, February 1991
After Operation Desert Storm, the army did not see major combat operations for the remainder of the 1990s but did participate in a number of peacekeeping activities. In 1990 the Department of Defense issued guidance for 'rebalancing' after a review of the Total Force Policy,[52] but in 2004, USAF Air War College scholars concluded the guidance would reverse the Total Force Policy which is an 'essential ingredient to the successful application of military force'.[53]

21st century

U.S. Army Rangers taking part in a raid during an operation in Nahr-e Saraj, Afghanistan
On 11 September 2001, 53 Army civilians (47 employees and six contractors) and 22 soldiers were among the 125 victims killed in the Pentagon in a terrorist attack when American Airlines Flight 77 commandeered by five Al-Qaeda hijackers slammed into the western side of the building, as part of the September 11 attacks.[54] In response to the 11 September attacks and as part of the Global War on Terror, U.S. and NATO forces invaded Afghanistan in October 2001, displacing the Taliban government. The U.S. Army also led the combined U.S. and allied invasion of Iraq in 2003; it served as the primary source for ground forces with its ability to sustain short and long-term deployment operations. In the following years, the mission changed from conflict between regular militaries to counterinsurgency, resulting in the deaths of more than 4,000 U.S. service members (as of March 2008) and injuries to thousands more.[55][56] 23,813 insurgents were killed in Iraq between 2003 and 2011.[57]


U.S. Army soldiers with the 2nd Battalion, 327th Infantry Regiment, 101st Airborne Division returning fire during a firefight with Taliban forces in Barawala Kalay Valley in Kunar province, Afghanistan, March 2011
Until 2009, the army's chief modernization plan, its most ambitious since World War II,[58] was the Future Combat Systems program. In 2009, many systems were canceled, and the remaining were swept into the BCT modernization program.[59] By 2017, the Brigade Modernization project was completed and its headquarters, the Brigade Modernization Command, was renamed the Joint Modernization Command, or JMC.[60] In response to Budget sequestration in 2013, Army plans were to shrink to 1940 levels,[61] although actual Active-Army end-strengths were projected to fall to some 450,000 troops by the end of FY2017.[62][63] From 2016 to 2017, the Army retired hundreds of OH-58 Kiowa Warrior observation helicopters,[64] while retaining its Apache gunships.[65] The 2015 expenditure for Army research, development and acquisition changed from $32 billion projected in 2012 for FY15 to $21 billion for FY15 expected in 2014.[66]

Organization

Organization of the United States Army within the Department of Defense
Planning
By 2017, a task force was formed to address Army modernization,[67] which triggered shifts of units: CCDC, and ARCIC, from within Army Materiel Command (AMC), and Army Training and Doctrine Command (TRADOC), respectively, to a new Army Command (ACOM) in 2018.[68] The Army Futures Command (AFC), is a peer of FORSCOM, TRADOC, and AMC, the other ACOMs.[69] AFC's mission is modernization reform: to design hardware, as well as to work within the acquisition process which defines materiel for AMC. TRADOC's mission is to define the architecture and organization of the Army, and to train and supply soldiers to FORSCOM.[70]: minutes 2:30–15:00 [42] AFC's cross-functional teams (CFTs) are Futures Command's vehicle for sustainable reform of the acquisition process for the future.[71] In order to support the Army's modernization priorities, its FY2020 budget allocated $30 billion for the top six modernization priorities over the next five years.[72] The $30 billion came from $8 billion in cost avoidance and $22 billion in terminations.[72]

Army Components
See also: Structure of the United States Army

U.S. Army organization chart[73]
The task of organizing the U.S. Army commenced in 1775.[74] In the first one hundred years of its existence, the United States Army was maintained as a small peacetime force to man permanent forts and perform other non-wartime duties such as engineering and construction works. During times of war, the U.S. Army was augmented by the much larger United States Volunteers which were raised independently by various state governments. States also maintained full-time militias which could also be called into the service of the army.


Senior American commanders of the European theatre of World War II.
*Seated are (from left to right) Generals William H. Simpson, George S. Patton, Carl A. Spaatz, Dwight D. Eisenhower, Omar Bradley, Courtney H. Hodges, and Leonard T. Gerow
*standing are (from left to right) Generals Ralph F. Stearley, Hoyt Vandenberg, Walter Bedell Smith, Otto P. Weyland, and Richard E. Nugent
By the twentieth century, the U.S. Army had mobilized the U.S. Volunteers on four occasions during each of the major wars of the nineteenth century. During World War I, the 'National Army' was organized to fight the conflict, replacing the concept of U.S. Volunteers.[75] It was demobilized at the end of World War I, and was replaced by the Regular Army, the Organized Reserve Corps and the state militias. In the 1920s and 1930s, the 'career' soldiers were known as the 'Regular Army' with the 'Enlisted Reserve Corps' and 'Officer Reserve Corps' augmented to fill vacancies when needed.[76]

In 1941, the 'Army of the United States' was founded to fight World War II. The Regular Army, Army of the United States, the National Guard and Officer/Enlisted Reserve Corps (ORC and ERC) existed simultaneously. After World War II, the ORC and ERC were combined into the United States Army Reserve. The Army of the United States was re-established for the Korean War and Vietnam War and was demobilized upon the suspension of the draft.[76]

Currently, the Army is divided into the Regular Army, the Army Reserve and the Army National Guard.[75] Some states further maintain state defense forces, as a type of reserve to the National Guard, while all states maintain regulations for state militias.[77] State militias are both 'organized', meaning that they are armed forces usually part of the state defense forces, or 'unorganized' simply meaning that all able-bodied males may be eligible to be called into military service.

The U.S. Army is also divided into several branches and functional areas. Branches include officers, warrant officers, and enlisted Soldiers while functional areas consist of officers who are reclassified from their former branch into a functional area. However, officers continue to wear the branch insignia of their former branch in most cases, as functional areas do not generally have discrete insignia. Some branches, such as Special Forces, operate similarly to functional areas in that individuals may not join their ranks until having served in another Army branch. Careers in the Army can extend into cross-functional areas for officer,[78] warrant officer, enlisted, and civilian personnel.

U.S. Army branches and functional areas
Branch	Insignia and colors		Branch	Insignia and colors		Functional Area (FA)
Acquisition Corps (AC)			Air Defense Artillery (AD)			Information Network Engineering (FA 26)
Adjutant General's Corps (AG)
Includes Army Bands (AB)	 		Armor (AR)
Includes Cavalry (CV)	 		Information Operations (FA 30)
Aviation (AV)			Civil Affairs Corps (CA)			Strategic Intelligence (FA 34)
Chaplain Corps (CH)
  		Chemical Corps (CM)			Space Operations (FA 40)
Cyber Corps (CY)			Dental Corps (DC)			Public Affairs Officer (FA 46)
Corps of Engineers (EN)			Field Artillery (FA)			Academy Professor (FA 47)
Finance Corps (FI)			Infantry (IN)			Foreign Area Officer (FA 48)
Inspector General (IG)			Logistics (LG)			Operations Research/Systems Analysis (FA 49)
Judge Advocate General's Corps (JA)			Military Intelligence Corps (MI)			Force Management (FA 50)
Medical Corps (MC)			Medical Service Corps (MS)			Acquisition (FA 51)[78]
Military Police Corps (MP)			Army Nurse Corps (AN)			Simulation Operations (FA 57)
Psychological Operations (PO)			Medical Specialist Corps (SP)			Army Marketing (FA 58)[79]
Quartermaster Corps (QM)			Staff Specialist Corps (SS)
(USAR and ARNG only)			Health Services (FA 70)
Special Forces (SF)			Ordnance Corps (OD)			Laboratory Sciences (FA 71)
Veterinary Corps (VC)			Public Affairs (PA)			Preventive Medicine Sciences (FA 72)
Transportation Corps (TC)			Signal Corps (SC)			Behavioral Sciences (FA 73)
Special branch insignias (for some unique duty assignments)
National Guard Bureau (NGB)			General Staff			U.S. Military Academy Staff
Chaplain Candidate			Officer Candidate			Warrant Officer Candidate
Aide-de-camp
               		Senior Enlisted Advisor (SEA)

Before 1933, members of the Army National Guard were considered state militia until they were mobilized into the U.S. Army, typically on the onset of war. Since the 1933 amendment to the National Defense Act of 1916, all Army National Guard soldiers have held dual status. They serve as National Guardsmen under the authority of the governor of their state or territory and as reserve members of the U.S. Army under the authority of the president, in the Army National Guard of the United States.

Since the adoption of the total force policy, in the aftermath of the Vietnam War, reserve component soldiers have taken a more active role in U.S. military operations. For example, Reserve and Guard units took part in the Gulf War, peacekeeping in Kosovo, Afghanistan, and the 2003 invasion of Iraq.

Army commands and army service component commands
 Headquarters, United States Department of the Army (HQDA):

Army Commands	Current commander	Location of headquarters[c]
 United States Army Forces Command (FORSCOM)[80]	GEN Andrew P. Poppas	Fort Liberty, North Carolina
 United States Army Futures Command (AFC)[81]	GEN James E. Rainey	Austin, Texas
 United States Army Materiel Command (AMC)[82]	LTG Christopher O. Mohan (acting)	Redstone Arsenal, Alabama
 United States Army Training and Doctrine Command (TRADOC)[83]	GEN Gary M. Brito	Fort Eustis, Virginia
Army Service Component Commands	Current commander	Location of headquarters
 United States Army Central (ARCENT)/Third Army[84]	LTG Patrick D. Frank	Shaw Air Force Base, South Carolina
 United States Army Europe and Africa/Seventh Army	GEN Darryl A. Williams[85]	Clay Kaserne, Wiesbaden, Germany
 United States Army North (ARNORTH)/Fifth Army[86]	LTG John R. Evans Jr.	Joint Base San Antonio, Texas
 United States Army Pacific (USARPAC)[87]	GEN Charles A. Flynn	Fort Shafter, Hawaii
 United States Army South (ARSOUTH)/Sixth Army[88]	MG William L. Thigpen	Joint Base San Antonio, Texas
 Military Surface Deployment and Distribution Command (SDDC)[89]	MG Gavin A. Lawrence	Scott AFB, Illinois
 United States Army Cyber Command (ARCYBER)[90][91][92]	LTG Maria B. Barrett	Fort Eisenhower, Georgia
 United States Army Space and Missile Defense Command/United States Army Forces Strategic Command (USASMDC/ARSTRAT)[93]	LTG Daniel L. Karbler	Redstone Arsenal, Alabama
 United States Army Special Operations Command (USASOC)[94]	LTG Jonathan P. Braga	Fort Liberty, North Carolina
Operational Force Headquarters	Current commander	Location of headquarters
 Eighth Army (EUSA)[95]	LTG Willard M. Burleson III	Camp Humphreys, South Korea
Direct reporting units	Current commander	Location of headquarters
 Arlington National Cemetery and Soldiers' and Airmen's Home National Cemetery[96]	Katharine Kelley[97] (civilian)	Arlington County, Virginia
Civilian Protection Center of Excellence[98]	Michael McNerney	Arlington County, Virginia
United States Army Joint Counter-Small Unmanned Aircraft Systems Office[99]		Arlington County, Virginia
 Military Postal Service Agency[100]		Arlington County, Virginia
 United States Army Acquisition Support Center (USAASC)[101]	Craig A. Spisak[102] (civilian)	Fort Belvoir, Virginia
 United States Army Civilian Human Resources Agency (CHRA)[103]	Carol Burton[104] (civilian)	Aberdeen Proving Ground, Maryland
 United States Army Corps of Engineers (USACE)	LTG Scott A. Spellmon[105]	Washington, D.C.
 United States Army Corrections Command (ACC)[106]	BG Duane Miller	Arlington County, Virginia
 United States Army Criminal Investigation Division (USACID)	Gregory D. Ford	Quantico, Virginia
 United States Army Human Resources Command (HRC)[107]	MG Thomas R. Drew	Fort Knox, Kentucky
 United States Army Intelligence and Security Command (INSCOM)	MG Timothy D. Brown	Fort Belvoir, Virginia
 United States Army Medical Command (MEDCOM)	LTG Mary V. Krueger	Joint Base San Antonio, Texas
 United States Army Military District of Washington (MDW)	MG Trevor J. Bredenkamp	Fort Lesley J. McNair, Washington, D.C.
 United States Army Recruiting Command (USAREC)[108]	MG Johnny K. Davis[109]	Fort Knox, Kentucky
 United States Army Test and Evaluation Command (ATEC)	MG James J. Gallivan[110]	Aberdeen Proving Ground, Maryland
 United States Army War College (AWC)[111]	MG David C. Hill	Carlisle, Pennsylvania
 United States Military Academy (USMA)	LTG Steven W. Gilland	West Point, New York
Source: U.S. Army organization[112]
Structure
Main article: Structure of the United States Army
See also: Transformation of the United States Army
See also: Reorganization plan of the United States Army

U.S. Army soldiers of the 1st Battalion, 175th Infantry Regiment, Maryland Army National Guard conducting an urban cordon and search exercise as part of the army readiness and training evaluation program in the mock city of Balad at Fort Dix, New Jersey

U.S. soldiers from the 6th Infantry Regiment taking up positions on a street corner during a foot patrol in Ramadi, Iraq

The 1st Cavalry Division's combat aviation brigade performing a mock charge with the horse detachment

U.S. Army Special Forces soldiers from the 3rd Special Forces Group patrolling a field in the Gulistan district of Farah, Afghanistan
See Structure of the United States Army for a detailed treatment of the history, components, administrative and operational structure and the branches and functional areas of the Army.

The U.S. Army is made up of three components: the active component, the Regular Army; and two reserve components, the Army National Guard and the Army Reserve. Both reserve components are primarily composed of part-time soldiers who train once a month – known as battle assemblies or unit training assemblies (UTAs) – and conduct two to three weeks of annual training each year. Both the Regular Army and the Army Reserve are organized under Title 10 of the United States Code, while the National Guard is organized under Title 32. While the Army National Guard is organized, trained and equipped as a component of the U.S. Army, when it is not in federal service it is under the command of individual state and territorial governors. However, the District of Columbia National Guard reports to the U.S. president, not the district's mayor, even when not federalized. Any or all of the National Guard can be federalized by presidential order and against the governor's wishes.[113]

The U.S. Army is led by a civilian secretary of the Army, who has the statutory authority to conduct all the affairs of the army under the authority, direction and control of the secretary of defense.[114] The chief of staff of the Army, who is the highest-ranked military officer in the army, serves as the principal military adviser and executive agent for the secretary of the Army, i.e., its service chief; and as a member of the Joint Chiefs of Staff, a body composed of the service chiefs from each of the four military services belonging to the Department of Defense who advise the president of the United States, the secretary of defense and the National Security Council on operational military matters, under the guidance of the chairman and vice chairman of the Joint Chiefs of Staff.[115][116] In 1986, the Goldwater–Nichols Act mandated that operational control of the services follows a chain of command from the president to the secretary of defense directly to the unified combatant commanders, who have control of all armed forces units in their geographic or function area of responsibility, thus the secretaries of the military departments (and their respective service chiefs underneath them) only have the responsibility to organize, train and equip their service components. The army provides trained forces to the combatant commanders for use as directed by the secretary of defense.[117]

By 2013, the army shifted to six geographical commands that align with the six geographical unified combatant commands (CCMD):

United States Army Central headquartered at Shaw Air Force Base, South Carolina
United States Army North headquartered at Fort Sam Houston, Texas
United States Army South headquartered at Fort Sam Houston, Texas
United States Army Europe and Africa headquartered at Clay Kaserne, Wiesbaden, Germany
United States Army Pacific headquartered at Fort Shafter, Hawaii
The army also transformed its base unit from divisions to brigades. Division lineage will be retained, but the divisional headquarters will be able to command any brigade, not just brigades that carry their divisional lineage. The central part of this plan is that each brigade will be modular, i.e., all brigades of the same type will be exactly the same and thus any brigade can be commanded by any division. As specified before the 2013 end-strength re-definitions, the three major types of brigade combat teams are:

Armored brigades, with a strength of 4,743 troops as of 2014.
Stryker brigades, with a strength of 4,500 troops as of 2014.
Infantry brigades, with a strength of 4,413 troops as of 2014.
In addition, there are combat support and service support modular brigades. Combat support brigades include aviation (CAB) brigades, which will come in heavy and light varieties, fires (artillery) brigades (now transforms to division artillery) and expeditionary military intelligence brigades. Combat service support brigades include sustainment brigades and come in several varieties and serve the standard support role in an army.

Combat maneuver organizations
To track the effects of the 2018 budget cuts, see Transformation of the United States Army#Divisions and brigades
The U.S. Army's conventional combat capability currently consists of 11 active divisions and 1 deployable division headquarters (7th Infantry Division) as well as several independent maneuver units.

From 2013 through 2017, the Army sustained organizational and end-strength reductions after several years of growth. In June 2013, the Army announced plans to downsize to 32 active brigade combat teams by 2015 to match a reduction in active-duty strength to 490,000 soldiers. Army chief of staff Raymond Odierno projected that the Army was to shrink to '450,000 in the active component, 335,000 in the National Guard and 195,000 in U.S. Army Reserve' by 2018.[118] However, this plan was scrapped by the incoming Trump administration, with subsequent plans to expand the Army by 16,000 soldiers to a total of 476,000 by October 2017. The National Guard and the Army Reserve will see a smaller expansion.[119][120]

The Army's maneuver organization was most recently altered by the reorganization of United States Army Alaska into the 11th Airborne Division, transferring the 1st and 4th Brigade Combat Teams of the 25th Infantry Division under a separate operational headquarters to reflect the brigades' distinct, Arctic-oriented mission. As part of the reorganization, the 1–11 (formerly 1–25) Stryker Brigade Combat Team will reorganize as an Infantry Brigade Combat Team.[121] Following this transition, the active component BCTs will number 11 Armored brigades, 6 Stryker brigades, and 14 Infantry brigades.

Within the Army National Guard and United States Army Reserve, there are a further eight divisions, 27 brigade combat teams, additional combat support and combat service support brigades, and independent cavalry, infantry, artillery, aviation, engineer and support battalions. The Army Reserve in particular provides virtually all psychological operations and civil affairs units.

 United States Army Forces Command (FORSCOM)

Direct reporting units	Current commander	Location of headquarters[c]
 I Corps	LTG Xavier T. Brunson	Joint Base Lewis-McChord, Washington
 III Armored Corps	LTG Sean Bernabe	Fort Cavazos, Texas
 V Corps	LTG John S. Kolasheski	Fort Knox, Kentucky
 XVIII Airborne Corps	LTG Christopher T. Donahue	Fort Liberty, North Carolina
 First Army[123]	MG Mark Landes Acting	Rock Island Arsenal, Illinois
 U.S. Army Reserve Command[124]	LTG Jody J. Daniels	Fort Liberty, North Carolina
 Security Force Assistance Command	MG Donn H. Hill	Fort Liberty, North Carolina
 20th CBRNE Command	BG Daryl O. Hood	Aberdeen Proving Ground, Maryland
 32nd Army Air and Missile Defense Command	BG David F. Stewart	Fort Bliss, Texas
 U.S. Army Air Traffic Services Command	COL Jason T. Cook	Fort Novosel, Alabama
Active combat maneuver units
Name	Headquarters	Subunits	Subordinate to
1st Armored Division	Fort Bliss, Texas	3 armored BCTs (ABCTs),[125] 1 Division Artillery (DIVARTY), 1 Combat Aviation Brigade (CAB), and 1 sustainment brigade	III Corps
1st Cavalry Division	Fort Cavazos, Texas	3 armored BCTs, 1 DIVARTY, 1 CAB, and 1 sustainment brigade	III Corps
 1st Infantry Division	Fort Riley, Kansas	2 armored BCTs, 1 DIVARTY, 1 CAB, and 1 sustainment brigade	III Corps
2nd Infantry Division	Camp Humphreys, South Korea
Joint Base Lewis–McChord, Washington	2 Stryker BCTs, 1 mechanized brigade from the ROK Army,[126] 1 DIVARTY (under administrative control of 7th ID), 1 sustainment brigade, and a stateside Stryker BCT from another active division that is rotated in on a regular basis.	I Corps (CONUS)
Eighth Army (OCONUS)
2nd Cavalry Regiment	Rose Barracks, Vilseck, Germany	4 Stryker squadrons, 1 engineer squadron, 1 fires squadron, and 1 support squadron	U.S. Army Europe and Africa
3rd Infantry Division	Fort Stewart, Georgia	2 armored BCT, 1 DIVARTY, 1 CAB, and 1 sustainment brigade as well as the 48th Infantry BCT of the Georgia Army National Guard	XVIII Airborne Corps
3rd Cavalry Regiment	Fort Cavazos, Texas	4 Stryker squadrons, 1 fires squadron, 1 engineer squadron, and 1 support squadron (overseen by the 1st Cavalry Division)[127]	III Corps
4th Infantry Division	Fort Carson, Colorado	2 Stryker BCT, 1 armored BCT, DIVARTY, 1 CAB, and 1 sustainment brigade	III Corps
10th Mountain Division	Fort Drum, New York	3 infantry BCTs, 1 DIVARTY, 1 CAB, and 1 sustainment brigade	XVIII Airborne Corps
11th Airborne Division	Joint Base Elmendorf–Richardson, Alaska	1 airborne infantry BCT, 1 infantry BCT, 2 attached aviation battalions, and 1 sustainment battalion	I Corps
25th Infantry Division	Schofield Barracks, Hawaii	2 infantry BCTs, 1 DIVARTY, 1 CAB, and 1 sustainment brigade	I Corps
82nd Airborne Division	Fort Liberty, North Carolina	3 airborne infantry BCTs, 1 airborne DIVARTY, 1 airborne CAB, and 1 airborne sustainment brigade	XVIII Airborne Corps
101st Airborne Division (Air Assault)	Fort Campbell, Kentucky	3 infantry BCTs, 1 DIVARTY, 1 CAB, and 1 sustainment brigade	XVIII Airborne Corps
173rd Airborne Brigade	Camp Ederle, Vicenza, Italy	3 airborne infantry battalions (including 1st Battalion, 143rd Infantry Regiment of the Texas and Rhode Island Army National Guard), 1 airborne field artillery battalion, 1 airborne cavalry squadron, 1 airborne engineer battalion,[128] and 1 airborne support battalion	U.S. Army Europe and Africa
 Combat maneuver units under the Army National Guard until federalized
Name	Locations	Subunits
28th Infantry Division	Pennsylvania, Ohio and Maryland	2nd Infantry BCT, 56th Stryker BCT, 28th CAB,  55th Maneuver Enhancement Brigade (MEB),[129] and the 28th Infantry Division Sustainment Brigade (SB)
29th Infantry Division	Virginia, Maryland, North Carolina and Florida	 30th Armored BCT,  53rd Infantry BCT, 116th Infantry BCT, 29th CAB,  142nd Field Artillery Regiment, 29th Infantry Division SB, and the  226th MEB[130]
34th Infantry Division	Minnesota, Wisconsin, Iowa and Idaho	1st Armored BCT, 2nd Infantry BCT,  32nd Infantry BCT,  116th Cavalry BCT,  115th Field Artillery Brigade, 34th CAB, 34th Infantry Division SB, and the  157th MEB
35th Infantry Division	Kansas, Missouri, Illinois, Oklahoma, Georgia, Arkansas, and Nebraska	 33rd Infantry BCT,  39th Infantry BCT,  45th Infantry BCT,  130th Field Artillery Brigade, 35th CAB, and the  67th MEB
36th Infantry Division	Texas, Louisiana and Mississippi	56th Infantry BCT, 72nd Infantry BCT,  256th Infantry BCT,  155th Armored BCT,  278th Armored Cavalry Regiment, 36th CAB, 36th Infantry Division SB, and the  136th MEB
38th Infantry Division	Indiana, Michigan, Ohio and Tennessee	 37th Infantry BCT,  76th Infantry BCT,  138th Field Artillery Brigade, 38th CAB, 38th Infantry Division SB, and the  149th MEB
40th Infantry Division	Arizona, California, Hawaii, Oregon, and Washington	 29th Infantry BCT,  41st Infantry BCT,  79th Infantry BCT,  81st Stryker BCT, 40th CAB, and the 40th Infantry Division SB
42nd Infantry Division	Connecticut, Maine, Maryland, Massachusetts, New Hampshire, New Jersey, New York, Rhode Island, and Vermont	 27th Infantry BCT,  44th Infantry BCT,  86th Infantry BCT (Mountain),  197th Field Artillery Brigade, 42nd CAB, 42nd Infantry Division SB, and the  26th MEB
For a description of U.S. Army tactical organizational structure, see: a U.S. context and also a global context.

Medical Department
Main article: Army Medical Department (United States)
The United States Army Medical Department (AMEDD), formerly the Army Medical Service (AMS), is the primary healthcare organization of the United States Army and is led by the Surgeon General of the United States Army (TSG), a three-star lieutenant general, who (by policy) also serves as the Commanding General, United States Army Medical Command (MEDCOM). TSG is assisted by a Deputy Surgeon General and a full staff, the Office of the Surgeon General (OTSG). The incumbent Surgeon General is Lieutenant General Mary K. Izaguirre (since January 25, 2024).

AMEDD encompasses the Army's six non-combat, medical-focused specialty branches (or 'Corps'), these branches are: the Medical Corps, Nurse Corps, Dental Corps, Veterinary Corps, Medical Specialist Corps, Medical Specialist Corps. Each of these branches is headed by a Corps Chief that reports directly to the Surgeon General.

Special operations forces
Main article: Army Special Operations Command
 United States Army Special Operations Command (Airborne) (USASOC):[131]

Name	Headquarters[c]	Structure and purpose
1st Special Forces Command	Fort Liberty (formerly Bragg), North Carolina	Manages seven special forces groups designed to deploy and execute nine doctrinal missions: unconventional warfare, foreign internal defense, direct action, counter-insurgency, special reconnaissance, counter-terrorism, information operations, counterproliferation of weapon of mass destruction, and security force assistance. The command also manages two psychological operations groups—tasked to work with foreign nations to induce or reinforce behavior favorable to U.S. objectives—a civil affairs brigade—that enables military commanders and U.S. ambassadors to improve relationships with various stakeholders via five battalions—and a sustainment brigade—that provides combat service support and combat health support units via three distinct battalions.
Army Special Operations Aviation Command	Fort Liberty, North Carolina	Commands, organizes, mans, trains, resources, and equips Army special operations aviation units to provide responsive, special operations aviation support to special operations forces consisting of five units, including the 160th Special Operations Aviation Regiment (Airborne).
75th Ranger Regiment	Fort Moore (formerly Benning), Georgia	In addition to a regimental headquarters, a special troops battalion, and a military intelligence battalion, the 75th Ranger Regiment has three maneuver battalions of elite airborne infantry specializing in large-scale, joint forcible entry operations and precision targeting raids. Additional capabilities include special reconnaissance, air assault, and direct action raids seizing key terrain such as airfields, destroying or securing strategic facilities, and capturing or killing enemies of the Nation. The Regiment also helps develop the equipment, technologies, training, and readiness that bridge the gap between special operations and traditional combat maneuver organizations.
John F. Kennedy Special Warfare Center and School	Fort Liberty, North Carolina	Selects and trains special forces, civil affairs, and psychological operations soldiers consisting of two groups and other various training units and offices.
1st Special Forces Operational Detachment-Delta	Fort Liberty, North Carolina	Commonly referred to as Delta Force, Combat Applications Group (CAG), 'The Unit', Army Compartmented Element (ACE), or Task Force Green, SFOD–D is the U.S. Army's Tier 1 Special Mission Unit tasked with performing the most complex, classified, and dangerous missions directed by the National Command Authority. Under the control of Joint Special Operations Command, SFOD–D specializes in hostage rescue, counter-terrorism, direct action, and special reconnaissance against high-value targets via eight squadrons: four assault, one aviation, one clandestine, one combat support, and one nuclear disposal.[132][133]
Personnel
See also: List of ranks used by the United States Army
The Army's Talent Management Task Force (TMTF) has deployed IPPS-A,[134] the Integrated Personnel and Pay System - Army, an app which serves the National Guard, and on 17 January 2023 the Army Reserve and Active Army.[135] Soldiers were reminded to update their information using the legacy systems to keep their payroll and personnel information current by December 2021. IPPS-A is the Human Resources system for the Army, is now available for download for Android, or the Apple store.[136] It will be used for future promotions and other personnel decisions. Among the changes are:

BCAP, the Battalion Commander Assessment Program. In January 2020, over 800 majors and lieutenant colonels from all over the Army converged on Fort Knox to take part in a five-day program to select the next battalion commanders for the Army (beginning in FY2021). This process replaces the former selection process which was based solely on rank and individual reviews of past performance. From now on, more consideration will be given to an individual officer's personal preference, as part of 25 other selection criteria.[137] 'Promotion boards will now be able to see almost all substantiated adverse information'.[138] The promotion boards will be able to see anything in an officer's human resource record. Officers are encouraged to become familiar with their human resource record, and to file rebuttals to adverse information.[138]
Depending on the success of this initiative, other assessment programs could be instituted as well, for promotion to sergeants major,[139] and for assessment of colonels for command.[140]
Below are the U.S. Army ranks authorized for use today and their equivalent NATO designations. Although no living officer currently holds the rank of General of the Army, it is still authorized by Congress for use in wartime.

Officers
Main article: United States Army officer rank insignia
There are several paths to becoming a commissioned officer[141] including the United States Military Academy, Reserve Officers' Training Corps, Officer Candidate School, and direct commissioning. Regardless of which road an officer takes, the insignia are the same. Certain professions including physicians, pharmacists, nurses, lawyers and chaplains are commissioned directly into the Army.

Most army commissioned officers (those who are generalists)[142] are promoted based on an 'up or out' system. A more flexible talent management process is underway.[142] The Defense Officer Personnel Management Act of 1980 establishes rules for the timing of promotions and limits the number of officers that can serve at any given time.

Army regulations call for addressing all personnel with the rank of general as 'General (last name)' regardless of the number of stars. Likewise, both colonels and lieutenant colonels are addressed as 'Colonel (last name)' and first and second lieutenants as 'Lieutenant (last name)'.[143]

US DoD
pay grade	Special grade[d]	O-10	O-9	O-8	O-7	O-6	O-5	O-4	O-3	O-2	O-1
NATO code	OF-10	OF-9	OF-8	OF-7	OF-6	OF-5	OF-4	OF-3	OF-2	OF-1
Insignia
Army Green Service Uniform
Title	General of the Army	General	Lieutenant general	Major general	Brigadier general	Colonel	Lieutenant colonel	Major	Captain	First lieutenant	Second lieutenant
Abbreviation	GA	GEN	LTG	MG	BG	COL	LTC	MAJ	CPT	1LT	2LT
Warrant officers
Main article: Warrant officer (United States)
Warrant officers[141] are single track, specialty officers with subject matter expertise in a particular area. They are initially appointed as warrant officers (in the rank of WO1) by the secretary of the Army, but receive their commission upon promotion to chief warrant officer two (CW2).

By regulation, warrant officers are addressed as 'Mr. (last name)' or 'Ms. (last name)' by senior officers and as 'sir' or 'ma'am' by all enlisted personnel.[143] However, many personnel address warrant officers as 'Chief (last name)' within their units regardless of rank.

US DoD pay grade	W-5	W-4	W-3	W-2	W-1
NATO code	WO-5	WO-4	WO-3	WO-2	WO-1
Insignia
Army Green Service Uniform
Title	Chief warrant officer 5	Chief warrant officer 4	Chief warrant officer 3	Chief warrant officer 2	Warrant officer 1
Abbreviation	CW5	CW4	CW3	CW2	WO1
Enlisted personnel
Main article: United States Army enlisted rank insignia
See also: Enlisted rank
Sergeants and corporals are referred to as NCOs, short for non-commissioned officers.[141][144] This distinguishes corporals from the more numerous specialists who have the same pay grade but do not exercise leadership responsibilities. Beginning in 2021, all corporals will be required to conduct structured self-development for the NCO ranks, completing the basic leader course (BLC), or else be laterally assigned as specialists. Specialists who have completed BLC and who have been recommended for promotion will be permitted to wear corporal rank before their recommended promotion as NCOs.[145]

Privates and privates first class (E3) are addressed as 'Private (last name)', specialists as 'Specialist (last name)', corporals as 'Corporal (last name)' and sergeants, staff sergeants, sergeants first class and master sergeants all as 'Sergeant (last name)'. First sergeants are addressed as 'First Sergeant (last name)' and sergeants major and command sergeants major are addressed as 'Sergeant Major (last name)'.[143]

US DoD
pay grade	Special	E-9	E-8	E-7	E-6	E-5	E-4	E-3	E-2	E-1
NATO code	OR-9	OR-8	OR-7	OR-6	OR-5	OR-4	OR-3	OR-2	OR-1
Uniform insignia														No insignia
Title	Senior Enlisted Advisor to the Chairman	Sergeant Major of the Army	Command sergeant major	Sergeant major	First sergeant	Master sergeant	Sergeant first class	Staff sergeant	Sergeant	Corporal	Specialist	Private first class	Private	Private
Abbreviation	SEAC	SMA	CSM	SGM	1SG[e]	MSG	SFC	SSG	SGT	CPL	SPC[f]	PFC	PV2[g]	PV1
Training

U.S. Army Rangers practicing fast roping techniques from an MH-47 during an exercise at Fort Liberty
Training in the U.S. Army is generally divided into two categories – individual and collective. Because of COVID-19 precautions, the first two weeks of basic training — not including processing and out-processing – incorporate social distancing and indoor desk-oriented training. Once the recruits have tested negative for COVID-19 for two weeks, the remaining 8 weeks follow the traditional activities for most recruits,[147] followed by Advanced Individualized Training (AIT) where they receive training for their military occupational specialties (MOS). Some individual's MOSs range anywhere from 14 to 20 weeks of One Station Unit Training (OSUT), which combines Basic Training and AIT. The length of AIT school varies by the MOS. The length of time spent in AIT depends on the MOS of the soldier. Certain highly technical MOS training requires many months (e.g., foreign language translators). Depending on the needs of the army, Basic Combat Training for combat arms soldiers is conducted at a number of locations, but two of the longest-running are the Armor School and the Infantry School, both at Fort Moore, Georgia. Sergeant Major of the Army Dailey notes that an infantrymen's pilot program for One Station Unit Training (OSUT) extends 8 weeks beyond Basic Training and AIT, to 22 weeks. The pilot, designed to boost infantry readiness ended in December 2018. The new Infantry OSUT covered the M240 machine gun as well as the M249 squad automatic weapon.[148] The redesigned Infantry OSUT started in 2019.[149][150] Depending on the result of the 2018 pilot, OSUTs could also extend training in other combat arms beyond the infantry.[149] One Station Unit Training will be extended to 22 weeks for Armor by Fiscal Year 2021.[23] Additional OSUTs are expanding to Cavalry, Engineer, and Military Police (MP) in the succeeding Fiscal Years.[151]

A new training assignment for junior officers was instituted, that they serve as platoon leaders for Basic Combat Training (BCT) platoons.[152] These lieutenants will assume many of the administrative, logistical, and day-to-day tasks formerly performed by the drill sergeants of those platoons and are expected to 'lead, train, and assist with maintaining and enhancing the morale, welfare and readiness' of the drill sergeants and their BCT platoons.[152] These lieutenants are also expected to stem any inappropriate behaviors they witness in their platoons, to free up the drill sergeants for training.[152]


A trainer with Company A, 1st Battalion 502nd Infantry Regiment, Task Force Strike, 101st Airborne Division assisting Iraqi army ranger students during a room clearing drill at Camp Taji, Iraq on 18 July 2016
The United States Army Combat Fitness Test (ACFT) was introduced in 2018 to 60 battalions spread throughout the Army.[153] The test and scoring system is the same for all soldiers, regardless of gender. It takes an hour to complete, including resting periods.[154] The ACFT supersedes the Army Physical Fitness Test (APFT),[155][156][157] as being more relevant to survival in combat.[153] Six events were determined to better predict which muscle groups of the body were adequately conditioned for combat actions:[154] three deadlifts,[158] a standing power throw of a ten-pound medicine ball,[159] hand-release pushups[160] (which replace the traditional pushup), a sprint/drag/carry 250 yard event,[161] three pull-ups with leg tucks (or a plank test in lieu of the leg tuck),[162][163] a mandatory rest period, and a two-mile run.[164] As of 1 October 2020 all soldiers from all three components (Regular Army, Reserve, and National Guard)[165] are subject to this test.[166][167] The ACFT now tests all soldiers in basic training as of October 2020. The ACFT became the official test of record 1 October 2020; before that day every Army unit was required to complete a diagnostic ACFT[168] (All Soldiers with valid APFT scores can use them until March 2022. The Holistic Health and Fitness (H2F) System is one way that soldiers can prepare.).[169][170][171] The ACFT movements directly translate to movements on the battlefield.[150] Following their basic and advanced training at the individual level, soldiers may choose to continue their training and apply for an 'additional skill identifier' (ASI). The ASI allows the army to take a wide-ranging MOS and focus it on a more specific MOS. For example, a combat medic, whose duties are to provide pre-hospital emergency treatment, may receive ASI training to become a cardiovascular specialist, a dialysis specialist, or even a licensed practical nurse. For commissioned officers, training includes pre-commissioning training, known as Basic Officer Leader Course A, either at USMA or via ROTC, or by completing OCS. After commissioning, officers undergo branch-specific training at the Basic Officer Leaders Course B, (formerly called Officer Basic Course), which varies in time and location according to their future assignments. Officers will continue to attend standardized training at different stages of their careers.[172]


U.S. Army soldiers familiarizing with the latest INSAS 1B1 during exercise Yudh Abhyas 2015
Collective training at the unit level takes place at the unit's assigned station, but the most intensive training at higher echelons is conducted at the three combat training centers (CTC); the National Training Center (NTC) at Fort Irwin, California, the Joint Readiness Training Center (JRTC) at Fort Johnson, Louisiana and the Joint Multinational Training Center (JMRC) at the Hohenfels Training Area in Hohenfels and Grafenwöhr,[173] Germany. ReARMM is the Army Force Generation process approved in 2020 to meet the need to continuously replenish forces for deployment, at unit level and for other echelons as required by the mission. Individual-level replenishment still requires training at a unit level, which is conducted at the continental U.S. (CONUS) replacement center (CRC) at Fort Bliss, in New Mexico and Texas before their individual deployment.[174]

Chief of Staff Milley notes that the Army is suboptimized for training in cold-weather regions, jungles, mountains, or urban areas where in contrast the Army does well when training for deserts or rolling terrain.[175]: minute 1:26:00  Post 9/11, Army unit-level training was for counter-insurgency (COIN); by 2014–2017, training had shifted to decisive action training.[176]

Equipment
Main article: List of equipment of the United States Army
The chief of staff of the Army has identified six modernization priorities, in order: artillery, ground vehicles, aircraft, network, air/missile defense, and soldier lethality.[177]

Weapons

A Lockheed Martin Terminal High Altitude Area Defense (THAAD) system used for ballistic missile protection
Individual weapons
The United States Army employs various weapons to provide light firepower at short ranges. The most common weapon type used by the army is the M4 carbine, a compact variant of the M16 rifle,[178] along with the 7.62×51mm variant of the FN SCAR for Army Rangers.Then the future weapon is the M7, which fires a 6.8mm round. The primary sidearm in the U.S. Army is the 9 mm M9 pistol; the M11 pistol is also used. Both handguns are to be replaced by the M17[179] through the Modular Handgun System program.[180] Soldiers are also equipped with various hand grenades, such as the M67 fragmentation grenade and M18 smoke grenade.

Many units are supplemented with a variety of specialized weapons, including the M249 SAW (Squad Automatic Weapon), to provide suppressive fire at the squad level.[181] Indirect fire is provided by the M320 grenade launcher. The M1014 Joint Service Combat Shotgun or the Mossberg 590 Shotgun are used for door breaching and close-quarters combat. The M14EBR is used by designated marksmen. Snipers use the M107 Long Range Sniper Rifle, the M2010 Enhanced Sniper Rifle and the M110 Semi-Automatic Sniper Rifle.

Crew-served weapons
The army employs various crew-served weapons to provide heavy firepower at ranges exceeding that of individual weapons.

The M240 is the U.S. Army's standard Medium Machine Gun.[182] The M2 heavy machine gun is generally used as a vehicle-mounted machine gun. In the same way, the 40 mm MK 19 grenade machine gun is mainly used by motorized units.[183]

The U.S. Army uses three types of mortar for indirect fire support when heavier artillery may not be appropriate or available. The smallest of these is the 60 mm M224, normally assigned at the infantry company level.[184] At the next higher echelon, infantry battalions are typically supported by a section of 81 mm M252 mortars.[185] The largest mortar in the army's inventory is the 120 mm M120/M121, usually employed by mechanized units.[186]

Fire support for light infantry units is provided by towed howitzers, including the 105 mm M119A1[187] and the 155 mm M777.[citation needed]

The U.S. Army utilizes a variety of direct-fire rockets and missiles to provide infantry with an Anti-Armor Capability. The AT4 is an unguided projectile that can destroy armor and bunkers at ranges up to 500 meters. The FIM-92 Stinger is a shoulder-launched, heat seeking anti-aircraft missile. The FGM-148 Javelin and BGM-71 TOW are anti-tank guided missiles.

Vehicles

A U.S. soldier on patrol in Iraq with the support of a Humvee vehicle
U.S. Army doctrine puts a premium on mechanized warfare. It fields the highest vehicle-to-soldier ratio in the world as of 2009.[188] The army's most common vehicle is the High Mobility Multipurpose Wheeled Vehicle (HMMWV), commonly called the Humvee, which is capable of serving as a cargo/troop carrier, weapons platform and ambulance, among many other roles.[189] While they operate a wide variety of combat support vehicles, one of the most common types centers on the family of HEMTT vehicles. The M1A2 Abrams is the army's main battle tank,[190] while the M2A3 Bradley is the standard infantry fighting vehicle.[191] Other vehicles include the Stryker,[192] the M113 armored personnel carrier[193] and multiple types of Mine Resistant Ambush Protected (MRAP) vehicles.


3rd Infantry Division soldiers manning an M1A1 Abrams in Iraq
The U.S. Army's principal artillery weapons are the M109A6 Paladin self-propelled howitzer[194] and the M270 Multiple Launch Rocket System (MLRS),[195] both mounted on tracked platforms and assigned to heavy mechanized units.

Aviation
While the United States Army Aviation Branch operates a few fixed-wing aircraft, it mainly operates several types of rotary-wing aircraft. These include the AH-64 Apache attack helicopter,[196] the UH-60 Black Hawk utility tactical transport helicopter[197] and the CH-47 Chinook heavy-lift transport helicopter.[198] Restructuring plans call for reduction of 750 aircraft and from seven to four types.[199] The Army is evaluating two fixed-wing aircraft demonstrators; ARES, and Artemis are under evaluation to replace the Guardrail ISR (Intelligence, surveillance and reconnaissance) aircraft.[200] Under the Johnson-McConnell agreement of 1966, the Army agreed to limit its fixed-wing aviation role to administrative mission support (light unarmed aircraft which cannot operate from forward positions). For UAVs, the Army is deploying at least one company of drone MQ-1C Gray Eagles to each Active Army division.[201]

Uniforms
Main article: Uniforms of the United States Army
The Army Combat Uniform (ACU) currently features a camouflage pattern known as Operational Camouflage Pattern (OCP); OCP replaced a pixel-based pattern known as Universal Camouflage Pattern (UCP) in 2019.

On 11 November 2018, the Army announced a new version of 'Army Greens' based on uniforms worn during World War II that will become the standard garrison service uniform.[202] The blue Army Service Uniform will remain as the dress uniform. The Army Greens are projected to be first fielded in the summer of 2020.[202][needs update]

The 2020 Army Greens uniform
The 2020 Army Greens uniform

An element of the 18th Infantry Regiment, wearing ASUs, representing the United States at the 2010 Moscow Victory Day Parade
An element of the 18th Infantry Regiment, wearing ASUs, representing the United States at the 2010 Moscow Victory Day Parade
Berets

The Ranger Honor Platoon marching in their tan berets and former service uniform
The beret flash of enlisted personnel displays their distinctive unit insignia (shown above). The U.S. Army's black beret is no longer worn with the ACU for garrison duty, having been permanently replaced with the patrol cap. After years of complaints that it was not suited well for most work conditions, Army Chief of Staff General Martin Dempsey eliminated it for wear with the ACU in June 2011. Soldiers who are currently in a unit in jump status still wear berets, whether the wearer is parachute-qualified or not (maroon beret), while members of Security Force Assistance Brigades (SFABs) wear brown berets. Members of the 75th Ranger Regiment and the Airborne and Ranger Training Brigade (tan beret) and Special Forces (rifle green beret) may wear it with the Army Service Uniform for non-ceremonial functions. Unit commanders may still direct the wear of patrol caps in these units in training environments or motor pools.

Tents
The Army has relied heavily on tents to provide the various facilities needed while on deployment (Force Provider Expeditionary (FPE)).[177]: p.146  The most common tent uses for the military are as temporary barracks (sleeping quarters), DFAC buildings (dining facilities),[203] forward operating bases (FOBs), after-action review (AAR), tactical operations center (TOC), morale, welfare and recreation (MWR) facilities, as well as security checkpoints. Furthermore, most of these tents are set up and operated through the support of Natick Soldier Systems Center. Each FPE contains billeting, latrines, showers, laundry and kitchen facilities for 50–150 Soldiers,[177]: p.146  and is stored in Army Prepositioned Stocks 1, 2, 4 and 5. This provisioning allows combatant commanders to position soldiers as required in their Area of Responsibility, within 24 to 48 hours.

The U.S. Army is beginning to use a more modern tent called the deployable rapid assembly shelter (DRASH). In 2008, DRASH became part of the Army's Standard Integrated Command Post System."""

TEXT2 = (
    "Count number in words only in this format: one, two, three, four, five, six, seven, eight, nine, teen, eleven, "
)

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

        # dataset = load_dataset("nanotron/needle_in_a_hay_stack_eval_dataset", split="train")
        # df = load_dataset("nanotron/needle_in_a_hay_stack_eval_dataset", split="train")

        # NOTE: filter out only samples with context_length is 32768
        dataset = dataset.filter(lambda x: x["context_length"] == 32768 and x["depth_percent"] == depth_percent)
        df = df.filter(lambda x: x["context_length"] == 32768 and x["depth_percent"] == depth_percent)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        df.set_format("pandas")
        df = df[:]

        # for i in range(10):
        #     texts = [dataset["prompt"][i]]
        #     needle = dataset["answer"][i]
        #     generate(args, model, tokenizer, texts, needle, parallel_context)

        responses = []
        from tqdm import tqdm

        for batch in tqdm(dataloader):
            print("--------------------------------------------------")
            print(f"target answer: {batch['answer']}")
            texts = batch["prompt"]
            # texts = batch["haystack_text"]
            # needle = batch["answer"]
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
