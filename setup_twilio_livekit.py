# setup_twilio_livekit.py

import os
import asyncio
from dotenv import load_dotenv
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from livekit import api

# The name of the agent we want to dispatch calls to.
# This should match the agent_name in your agent's main execution block.
AGENT_NAME = "livekit-tutorial-hugo"

async def main():
    """
    This script provides a one-stop setup for connecting Twilio and LiveKit.
    It provisions and configures all necessary resources on both platforms.
    """
    print("🚀 Twilio & LiveKit Full Telephony Setup")
    print("-" * 60)

    load_dotenv()

    # --- 1. Gather Credentials and User Input ---
    print("Loading credentials from .env file...")
    twilio_account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    twilio_auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    livekit_api_key = os.environ.get("LIVEKIT_API_KEY")
    livekit_api_secret = os.environ.get("LIVEKIT_API_SECRET")
    livekit_url = os.environ.get("LIVEKIT_URL")

    if not all([twilio_account_sid, twilio_auth_token, livekit_api_key, livekit_api_secret, livekit_url]):
        print("❌ Error: Ensure all required environment variables are set in your .env file.")
        print("Required: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_URL")
        return

    print("Please provide the following information:")
    base_name = input("Enter a base name for resources (e.g., 'my-agent'): ")
    phone_number = input("Enter your Twilio phone number in E.164 format (e.g., +15551234567): ")
    sip_username = input("Enter a NEW username for SIP authentication: ")
    sip_password = input("Enter a NEW secure password for SIP authentication: ")
    livekit_sip_uri_raw = input("Enter your LiveKit SIP URI (e.g., 3kxm9r7vbn4q.sip.livekit.cloud): ").strip()
    livekit_sip_uri = livekit_sip_uri_raw.replace("sip://", "").replace("sip:", "")

    if not all([base_name, phone_number, sip_username, sip_password]):
        print("❌ Error: All fields are required.")
        return

    # Instantiate API clients
    twilio_client = Client(twilio_account_sid, twilio_auth_token)
    lk_api = api.LiveKitAPI(api_key=livekit_api_key, api_secret=livekit_api_secret, url=livekit_url)

    try:
        # --- 2. LiveKit Inbound Setup ---
        print("\n[Step 1/5] Setting up LiveKit for INBOUND calls (trunk first)...")

        inbound_trunk_info = await lk_api.sip.create_sip_inbound_trunk(
            api.CreateSIPInboundTrunkRequest(
                trunk=api.SIPInboundTrunkInfo(
                    name=f"{base_name}-inbound",
                    numbers=[phone_number],
                )
            )
        )

        livekit_origination_url = f"sip:{inbound_trunk_info.sip_trunk_id}@sip.livekit.cloud"
        print(f"✅ LiveKit Inbound Trunk created. Origination URL: {livekit_origination_url}")

        # Now that we have the trunk ID, create a dispatch rule tied to this trunk.
        print("\n[Step 2/5] Creating Dispatch Rule bound to inbound trunk…")
        rule = api.SIPDispatchRule(
            dispatch_rule_individual=api.SIPDispatchRuleIndividual(room_prefix=f"{base_name}-")
        )
        room_config = api.RoomConfiguration(agents=[api.RoomAgentDispatch(agent_name=AGENT_NAME)])
        dispatch_rule_info = await lk_api.sip.create_sip_dispatch_rule(
            api.CreateSIPDispatchRuleRequest(
                name=f"{base_name}-rule",
                rule=rule,
                room_config=room_config,
            )
        )
        print(f"✅ Dispatch Rule created. ID: {dispatch_rule_info.sip_dispatch_rule_id}")

        # --- 3. Twilio Trunk & Credential Setup ---
        print("\n[Step 3/5] Creating and configuring Twilio SIP Trunk...")
        twilio_trunk = twilio_client.trunking.v1.trunks.create(
            friendly_name=f"{base_name}-trunk",
            domain_name=f"{base_name}.pstn.twilio.com"  # Termination SIP Domain must end with pstn.twilio.com
        )

        # Sometimes domain_name is not immediately available; try refetching.
        twilio_termination_uri = twilio_trunk.domain_name
        if not twilio_termination_uri:
            import time
            time.sleep(2)
            twilio_termination_uri = twilio_client.trunking.v1.trunks(twilio_trunk.sid).fetch().domain_name

        if not twilio_termination_uri:
            twilio_termination_uri = input("Twilio did not return a trunk domain automatically. Enter the Termination SIP domain (e.g., your-trunk.pstn.twilio.com): ").strip()

        print(f"✅ Twilio Trunk created. SID: {twilio_trunk.sid}, Termination URI: {twilio_termination_uri}")

        print("\n[Step 4/5] Setting up Twilio Credential List...")
        cred_list = twilio_client.sip.credential_lists.create(friendly_name=f"{base_name}-creds")
        
        twilio_client.sip.credential_lists(cred_list.sid).credentials.create(
            username=sip_username, password=sip_password
        )
        
        twilio_client.trunking.v1.trunks(twilio_trunk.sid).credentials_lists.create(credential_list_sid=cred_list.sid)
        print(f"✅ Twilio Credential List created and associated with Trunk.")

        # --- 4. LiveKit Outbound Setup ---
        print("\n[Step 5/5] Setting up LiveKit for OUTBOUND calls...")
        outbound_trunk_info = await lk_api.sip.create_sip_outbound_trunk(
            api.CreateSIPOutboundTrunkRequest(
                trunk=api.SIPOutboundTrunkInfo(
                    name=f"{base_name}-outbound",
                    address=twilio_termination_uri,
                    numbers=[phone_number],
                    auth_username=sip_username,
                    auth_password=sip_password,
                    transport=api.SIPTransport.SIP_TRANSPORT_TLS,
                )
            )
        )
        livekit_outbound_trunk_id = outbound_trunk_info.sip_trunk_id
        print(f"✅ LiveKit Outbound Trunk created. ID: {livekit_outbound_trunk_id}")

        # --- 6. Final Twilio Configuration ---
        print("\n[Step 6/6] Connecting Twilio to LiveKit...")
        twilio_client.trunking.v1.trunks(twilio_trunk.sid).origination_urls.create(
            weight=1, priority=1, enabled=True,
            friendly_name=f"{base_name} LiveKit Origination",
            sip_url=f"sip:{livekit_sip_uri}"
        )
        
        incoming_phone_numbers = twilio_client.incoming_phone_numbers.list(phone_number=phone_number, limit=1)
        if not incoming_phone_numbers:
            raise Exception(f"Phone number {phone_number} not found in your Twilio account.")
        
        twilio_client.incoming_phone_numbers(incoming_phone_numbers[0].sid).update(trunk_sid=twilio_trunk.sid)
        print("✅ Twilio Trunk successfully linked to LiveKit and your phone number.")

        # --- Final Instructions ---
        print("\n" + "="*60)
        print("🎉 Full Telephony Setup Complete! 🎉")
        print("\nIMPORTANT: Please update your agent's code with the new LiveKit Outbound Trunk ID.")
        print(f"In your 'agent.py' file, find the 'create_sip_participant' call and replace the")
        print(f"'sip_trunk_id' with the following value:")
        print(f"\n    sip_trunk_id='{livekit_outbound_trunk_id}'\n")

    except TwilioRestException as e:
        print(f"\n❌ An error occurred with the Twilio API: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
    finally:
        if lk_api:
            await lk_api.aclose()

if __name__ == "__main__":
    asyncio.run(main()) 